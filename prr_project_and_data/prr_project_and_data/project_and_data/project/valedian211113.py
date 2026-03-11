import itertools as it
import logging
from copy import copy, deepcopy

import qutip as qt
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from qutils import sup_on_op
import miscellany as mcl
from miscellany import NO_VALUE

'''
TO DO:
    This implemtation of Xxx.load methods calls mcl.import_file a lot of times, which cause lots of 'duplicated' module objects are created.
        This is because mcl.import_file called on the same arguments creates different 'duplicated' module objects from the file.
        This behaviour differs from the import command.
    Also, the attributes names to be imported from these modules are hardcoded.
    Bad!
    Use a dictionary to put all the modules executed from files so that no duplicated object would be created,
        and a dictionary to tell Xxx.load method what attributes to use.
TO DO:
    In this module, all the qt.Qobj objects are assumed to be never muted during their lifetime.
        This is unsafe.
'''


class ChannelModel(object):
    '''
    merely a container
    '''

    def __init__(self,
                 hamiltonian_model=NO_VALUE,
                 collapse_op_model=NO_VALUE,
                 *,
                 make_copy=True
                 ):
        """
        Initialize an instance of ChannelModel.

        Args:
            hamiltonian_model: callable
                which maps (params: 1-D array of immutable objects) to a (hamiltonian: qutip.Qobj of type 'oper').
            collapse_op_model: callable
                which maps (params: 1-D array of immutable objects) to a list of (lindblad operators: qutip.Qobj of type 'oper').
            make_copy: bool
                When make_copy == True, the instance would use deepcopies of hamiltonian_model and collapse_op_model as its attributes. When it is False, the instance uses references instead.

        Notes:
            Because the Qobj dims of the hamiltonian and the lindblad operators cannot be inferred when an instance of ChannelModel is initialized, you should manually check whether the dims are compatible.
        """
        if hamiltonian_model is NO_VALUE and collapse_op_model is NO_VALUE:
            raise Exception(
                'at least one of hamiltonian_model and collapse_op_model should be provided')
        if make_copy:
            self.hamiltonian_model = deepcopy(hamiltonian_model)
            self.collapse_op_model = deepcopy(collapse_op_model)
        else:
            self.hamiltonian_model = hamiltonian_model
            self.collapse_op_model = collapse_op_model

    _subr_to_load = (
        'H',
        'C_list'
    )


class Controller(object):
    _set = {
        'times_list': 'set_times_list',
        'time_invariant': 'set_time_invariant',
        'adjustable': 'set_adjustable',
        'parameters': 'set_parameters',
        'interp_kwargs': 'set_interp_kwargs'
    }
    _get = {
        'times_list': 'get_times_list',
        'adjustable': 'get_adjustable_indice',
        'parameters': 'get_parameters',
        'parameters_shape': 'get_parameters_shape',
        'length_times_list': 'get_length_times_list'
    }

    class _Auxiliaries(object):
        def __init__(self, controller):
            self.length_times_list = len(controller._times_list)
            ini_t, fin_t = controller._times_list[0], controller._times_list[-1]
            self.time_interval = (ini_t, fin_t)
            self.time_interval_length = fin_t - ini_t
            self.parameter_shape = controller._parameters_list.shape
            self.parameter_length = self.parameter_shape[1]
            self.parameter_indice_set = set(range(self.parameter_length))

    def __init__(self,
                 times_list,
                 parameters_list,
                 *,
                 time_invariant=NO_VALUE,
                 adjustable=NO_VALUE,
                 make_copy=True,
                 **interpolation_kwargs
                 ):
        """
        Args:
            times_list: a 1-D numpy array of immutable objects
            parameters_list: a 2-D numpy array of immutable objects
                parameters_list.shape[0] should equals len(times_list).
            time_invariant: a list of integers
                No negtive index. No repeated element.
            adjustable: a list of integers
                No negtive index. No repeated element.
            make_copy: bool
            interpolation_kwargs: dict

        Notes:
            When an instance is initialized, time_invariant constrait will always be applied to the parameters_list even if some columns of it is set to be not adjustable.
            Only the intersection of time_invariant and adjustable really works as the constraits when another parameters_list is being set. Hence one should be careful that although -1 and len(parameters_list.shape[1]) refers to the same index, they are not considered to be the same when calculating the intersection of time_invariant and adjustable.
            It is highly recommended that time_invariant or adjustable has no repeated element, and all belong to set(range(parameters_list.shape[1])).
            However, no exception or warning would occur if time_invariant or adjustable didn't follow this recommendation. One should manually check that. Otherwise it's highly possible the codes don't work as intended.
        """
        if len(times_list) < 2:
            raise Exception('times_list should have at least two elements')
        if len(times_list) != parameters_list.shape[0]:
            raise Exception(
                'shapes of times_list and parameters_list not compatiable')

        self.set_times_list(times_list, make_copy=make_copy,
                            _do_refresh=False, _do_check=False)
        '''
        The order of the attributes being initialized is complicated because the 'set' functions for each attribute rely on some other attributes of the instance.
        '''
        self._time_invariant_true_indice = set()
        self.set_parameters(
            parameters_list,
            make_copy=make_copy,
            _ignore_constraits=True,
            _do_refresh=False,
            _do_check=False
        )
        self._refresh_aux()
        self.set_interp_kwargs(
            interpolation_kwargs,
            make_copy=make_copy,
            _do_refresh=False
        )
        if time_invariant is NO_VALUE:
            self.set_time_invariant(
                [], make_copy=False, _apply_constrait=False, _do_check=False)
        else:
            self.set_time_invariant(
                time_invariant, make_copy=make_copy, _apply_constrait=False)
        self._apply_time_invariant(obey_adjustable=False, do_refresh=True)
        if adjustable is NO_VALUE:
            self.set_adjustable(
                list(range(self._aux.parameter_length)), make_copy=False, _do_check=False)
        else:
            self.set_adjustable(adjustable, make_copy=make_copy)

    def safe_set(self, attr_name, new_value, *, make_copy=True):
        """
        Args:
            attr_name: str
                Among {'times_list', 'time_invariant', 'adjustable', 'parameters', 'interp_kwargs'}.

        Notes:
            Safe means that all constraits using this method
        """
        self.__getattribute__(self._set[attr_name])(
            new_value, make_copy=make_copy)

    def safe_get(self, attr_name):
        """
        Args:
            attr_name: str
                Among {'times_list', 'adjustable', 'parameters', 'parameters_shape', 'length_times_list'}.
        """
        self.__getattribute__(self._get[attr_name])()

    def set_times_list(self, times_list, *, make_copy=True, _do_refresh=True, _do_check=True):
        if _do_check and len(times_list) != self._aux.length_times_list:
            raise Exception(
                'shapes of times_list and parameters_list not compatiable')
        self._times_list = times_list.copy() if make_copy else times_list
        if _do_refresh:
            self._refresh_aux()
            self._refresh_interpolation()

    def set_time_invariant(self, time_invariant, *, make_copy=True, _apply_constrait=True, _do_check=True):
        if _do_check:
            idx_inv_set = set(time_invariant)
            if len(idx_inv_set) != len(time_invariant):
                raise Exception(
                    'time_invariant is not allowed to have duplicated elements')
            if not idx_inv_set.issubset(self._aux.parameter_indice_set):
                raise Exception(
                    'time_invariant is not allowed to have any element not in set(range(parameters_list.shape[1]))')
        self._time_invariant = time_invariant.copy() if make_copy else time_invariant
        if _apply_constrait:
            self._apply_time_invariant()

    def set_adjustable(self, adjustable, *, make_copy=True, _do_check=True):
        if _do_check:
            idx_adj_set = set(adjustable)
            if len(idx_adj_set) != len(adjustable):
                raise Exception(
                    'time_invariant is not allowed to have duplicated elements')
            if not idx_adj_set.issubset(self._aux.parameter_indice_set):
                raise Exception(
                    'time_invariant is not allowed to have any element not in set(range(parameters_list.shape[1]))')
        self._adjustable = adjustable.copy() if make_copy else adjustable

    def set_parameters(self, parameters_list, *, make_copy=True, _ignore_constraits=False, _do_refresh=True, _do_check=True):
        if _do_check and parameters_list.shape[0] != self._aux.length_times_list:
            raise Exception(
                'shapes of times_list and parameters_list not compatiable')
        if _ignore_constraits:
            self._parameters_list = parameters_list.copy() if make_copy else parameters_list
            self._time_invariant_true_indice.clear()
            if _do_refresh:
                self._refresh_interpolation()
        else:
            self._parameters_list[:, self._adjustable] = parameters_list[:, self._adjustable].copy(
            ) if make_copy else parameters_list[:, self._adjustable]
            self._time_invariant_true_indice.difference_update(
                self._adjustable)
            self._apply_time_invariant(do_refresh=_do_refresh)

    def _set_param_to_its_average(self, param_idx):
        interp_kwargs = self._interp_kwargs.copy()
        interp_kwargs['fill_value'] = self._interp_kwargs['fill_value'][param_idx]

        self._parameters_list[:, param_idx], _ = quad(
            interp1d(
                self._times_list,
                self._parameters_list[:, param_idx],
                **interp_kwargs
            ), *self._aux.time_interval,
            limit=100) / self._aux.time_interval_length

    def _apply_time_invariant(self, *, obey_adjustable=True, do_refresh=True):
        """
        Notes:
            Make sure self._interp_kwargs and self._aux is up-to-date before applying _time_invariant constraits.
            Make sure len(self._times_list) >= 2.
        """
        if obey_adjustable:
            inv_and_adj_set = set(
                self._time_invariant).intersection(self._adjustable)
            for idx in inv_and_adj_set.difference(self._time_invariant_true_indice):
                self._set_param_to_its_average(idx)
            self._time_invariant_true_indice.update(inv_and_adj_set)
        else:
            time_invariant_set = set(self._time_invariant)
            for idx in time_invariant_set.difference(self._time_invariant_true_indice):
                self._set_param_to_its_average(idx)
            self._time_invariant_true_indice.update(time_invariant_set)
        if do_refresh:
            self._refresh_interpolation()

    def set_interp_kwargs(self, interpolation_kwargs, *, make_copy=True, _do_refresh=True):
        """
        Notes:
            Make sure self._aux is up-to-date before setting interp_kwargs.
        """
        self._interp_kwargs = interpolation_kwargs.copy(
        ) if make_copy else interpolation_kwargs
        self._interp_kwargs['axis'] = 0
        self._interp_kwargs['copy'] = False
        if not 'kind' in self._interp_kwargs:
            if self._aux.length_times_list > 3:
                self._interp_kwargs['kind'] = 'cubic'
            else:
                self._interp_kwargs['kind'] = self._aux.length_times_list - 1
        if not 'bounds_error' in self._interp_kwargs:
            self._interp_kwargs['bounds_error'] = False
        if not 'fill_value' in self._interp_kwargs:
            self._interp_kwargs['fill_value'] = np.zeros(
                (self._aux.parameter_length,))
        if _do_refresh:
            self._refresh_interpolation()

    def _refresh_interpolation(self):
        self._interpolated_params = interp1d(
            self._times_list,
            self._parameters_list,
            **self._interp_kwargs
        ) if self._aux.length_times_list else lambda t: np.zeros((0,))

    def _refresh_aux(self):
        """
        Notes:
            Make sure self._parameters_list is initialized before refreshing _aux.
        """
        self._aux = self._Auxiliaries(self)

    def get_parameters(self, *, make_copy=True):
        return self._parameters_list.copy() if make_copy else self._parameters_list

    def get_parameters_shape(self):
        return self._aux.parameter_shape

    def get_parameter_length(self):
        return self._aux.parameter_length

    def get_first_parameters(self, *, make_copy=True):
        return self._parameters_list[0].copy() if make_copy else self._parameters_list[0]

    def get_times_list(self, *, make_copy=True):
        return self._times_list.copy() if make_copy else self._times_list

    def get_length_times_list(self):
        return self._aux.length_times_list

    def get_time_interval(self):
        return self._aux.time_interval

    def get_time_interval_length(self):
        return self._aux.time_interval_length

    def get_adjustable_indice(self, *, make_copy=True):
        return self._adjustable.copy() if make_copy else self._adjustable

    def is_time_invariant_for_all_params(self):
        return self._aux.parameter_length == len(self._time_invariant_true_indice)

    def copy(self):
        return copy(self)

    def __call__(self, time):
        return self._interpolated_params(time)


class ParameterizedLindbladChannel(object):
    class _Auxiliaries(object):
        def __init__(self, channel):
            """
            Notes:
                Don't assign any value to any attribute of instances of this class manually.
                All attributes are intended to be read only.
            """
            self.operator_structure = (
                channel._model.hamiltonian_model(
                    channel._controller.get_first_parameters(make_copy=False))
            ).dims
            self.super_operator_structure = [self.operator_structure] * 2
            self.hilbert_structure = self.operator_structure[0]
            self.hilbert_dimention = mcl.prod(self.hilbert_structure)

    class _InterResults(object):
        """
        Notes:
            Don't assign any value to any attribute of instances of this class manually.
            All attributes are intended to be read only.
        """

        def __init__(self, channel, *, delay_inter_solve=False):
            if delay_inter_solve:
                self.is_uptodate = False
            else:
                self.refresh(channel)

        def refresh(self, channel):
            if channel._controller.is_time_invariant_for_all_params():
                times_list = channel._controller.get_times_list(
                    make_copy=False)
                ini_time = times_list[0]
                ini_params = channel._controller.get_first_parameters()
                const_liouvillian = qt.liouvillian(
                    channel._model.hamiltonian_model(ini_params),
                    channel._model.collapse_op_model(ini_params)
                )
                try:
                    self.super_op_from_ini_list = [
                        ((time - ini_time) * const_liouvillian).expm()
                        for time in times_list
                    ]
                except Exception as e:
                    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S', filename='expm.log', filemode='a')
                    logging.error(e)
                    logging.error('\nliouv=%s' % const_liouvillian)
                try:
                    self.super_op_to_ini_list = [
                        (-(time - ini_time) * const_liouvillian).expm()
                        for time in times_list
                    ]
                except Exception as e:
                    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S', filename='inverse_expm.log', filemode='a')
                    logging.error(e)
                    logging.error('\nliouv=%s' % const_liouvillian)

            self.super_op_to_from_array = [
                        [op_f * op_i_inv for op_i_inv in self.super_op_to_ini_list]
                        for op_f in self.super_op_from_ini_list
                    ]
            self.is_uptodate = True

    def __init__(self,
                 model,
                 controller,
                 *,
                 make_copy=True,
                 delay_inter_solve=False
                 ):
        """
        Notes:
            Normally, it should be avoided that instances of ChannelModel are modified during runtime. They are copied just in case someone need to do that on purposes. And attributes of them should normally be immutable function objects, we use deepcopy to maintain the potential. The learning_subroutine attribute of the VariationalLearningPLC is in the same situation.
        """
        if make_copy:
            self._model = deepcopy(model)
            self._controller = deepcopy(controller)
        else:
            self._model = model
            self._controller = controller
        # TO DO: use different times_list for the channel and its controller
        self._aux = self._Auxiliaries(self)
        self._inter_results = self._InterResults(
            self, delay_inter_solve=delay_inter_solve)

    def refresh_inter_results(self):
        if not self._inter_results.is_uptodate:
            self._inter_results.refresh(self)

    def get_super_operator(self, *, make_copy=True):
        self.refresh_inter_results()
        return deepcopy(
                self._inter_results.super_op_from_ini_list[-1]
        ) if make_copy else self._inter_results.super_op_from_ini_list[-1]

    def get_super_operator_list(self, *, make_copy=True):
        self.refresh_inter_results()
        return deepcopy(
            self._inter_results.super_op_from_ini_list
        ) if make_copy else self._inter_results.super_op_from_ini_list

    def get_inversed_super_operator_list(self, *, make_copy=True):
        self.refresh_inter_results()
        try:
            return deepcopy(
                self._inter_results.super_op_to_ini_list
            ) if make_copy else self._inter_results.super_op_to_ini_list
        except Exception as e:
            logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S', filename='plc-inverse.log', filemode='a')
            logging.error(e)
            logging.error('self.super_op_from_ini_list[-1]=%s' % self.super_op_from_ini_list[-1])

    def get_super_operator_array(self, *, make_copy=True):
        self.refresh_inter_results()
        return deepcopy(
            self._inter_results.super_op_to_from_array
        ) if make_copy else self._inter_results.super_op_to_from_array

    def __call__(self, initial_dm):
        self.refresh_inter_results()
        return sup_on_op(
            self._inter_results.super_op_from_ini_list[-1],
            initial_dm
        )

    def __str__(self):
        self.refresh_inter_results()
        return str(self._inter_results.super_op_from_ini_list[-1])


class VariationalLearningPLC(ParameterizedLindbladChannel):
    class LearningSubroutines(object):
        '''merely a container'''
        # TO DO: replace with an automatic differetiation method

        def __init__(self,
                     pH_pp,
                     pC_pp_list,
                     dl_df=NO_VALUE,
                     pl_pp_channel_part=NO_VALUE,
                     pl_pp_joint_part=NO_VALUE,
                     loss=NO_VALUE,
                     *,
                     make_copy=True
                     ):
            """
            Args:
                pH_pp: callable
                    pH_pp([p0, p1, ...]) == [pH_pp0, pH_pp1, ...]
                pC_pp_list: callable
                    pC_pp_list([p0, p1, ...]) == [
                        [pC0_pp0, pC1_pp0, ...],
                        [pC0_pp1, pC1_pp1, ...],   # 已修改
                        ...
                    ]
                dl_df: callable
                    dl_df(dm_fin, any, delta_dm_fin) == delta_loss,
                    where dm_fin is calculated by applying the channel to a provided dm_ini,
                    and any, called element_for_calculate_desired_output in this script, is something provided paired with dm_ini.
                    Most probably, any is a desired dm_fin
                    to which the calculated one is the more similar, the better.
                pl_pp_channel_part: callable
                    pl_pp_channel_part(channel) == [
                        [pl_pp0(t=0), pl_pp1(t=0), ...],
                        [pl_pp0(t=1), pl_pp1(t=1), ...],
                        ...
                    ]
                pl_pp_joint_part: callable
                    pl_pp_joint_part(channel, dm_ini, any) == [
                        [pl_pp0(t=0), pl_pp1(t=0), ...],
                        [pl_pp0(t=1), pl_pp1(t=1), ...],
                        ...
                    ]
                    Most probably, any is a desired dm_fin
                    to which the calculated one is the more similar, the better.
                loss: any
                    Not used at all.
                make_copy: bool
            """
            if pl_pp_channel_part is NO_VALUE and dl_df is NO_VALUE:
                raise Exception(
                    'at least one of dl_df and pl_pp_channel_part subroutine should be provided')
            if make_copy:
                self.pH_pp = deepcopy(pH_pp)
                self.pC_pp_list = deepcopy(pC_pp_list)
                self.dl_df = deepcopy(dl_df)
                self.pl_pp_channel_part = deepcopy(pl_pp_channel_part)
                self.pl_pp_joint_part = deepcopy(pl_pp_joint_part)
                self.loss = deepcopy(loss)
            else:
                self.pH_pp = pH_pp
                self.pC_pp_list = pC_pp_list
                self.pl_pp_channel_part = pl_pp_channel_part
                self.dl_df = dl_df
                self.pl_pp_joint_part = pl_pp_joint_part
                self.loss = loss

        _subr_to_load = (
            'pH_pp',
            'pC_pp_list',
            'dl_df',
            'pl_pp_channel_part',
            'pl_pp_joint_part',
            'loss'
        )

        class Preset(object):
            class DlDpChannelPart(object):
                class SuperHilSchDistance(object):
                    def __init__(self, super_op_target, factor=0.01):
                        self._super_op_target = super_op_target
                        self._factor = factor

                    def __call__(self, learning_channel):
                        super_op_from_ini_list = learning_channel._inter_results.super_op_from_ini_list
                        super_op = super_op_from_ini_list[-1]
                        psuper_op_pp = [
                            [
                                op_to_fin * pLiou_ppi * op_from_ini
                                for pLiou_ppi in pLiou_pp
                            ]
                            for pLiou_pp, op_from_ini, op_to_fin in zip(
                                learning_channel._derivatives.pLiou_pp_on_times,
                                super_op_from_ini_list,
                                learning_channel._inter_results.super_op_to_from_array[-1]
                            )
                        ]

                        len_times_list = learning_channel._controller.get_length_times_list()
                        pl_pp = np.zeros((
                            len_times_list,
                            learning_channel._controller.get_parameter_length()
                        ))
                        for time_ind, param_ind in it.product(
                            range(len_times_list),
                            learning_channel._controller.get_adjustable_indice(
                                make_copy=False
                            )
                        ):  # TO DO: to optimize (replace the iteration with matrix caculation)
                            pl_pp[time_ind, param_ind] = (
                                (psuper_op_pp[time_ind][param_ind] -
                                 self._super_op_target).dag()
                                * (super_op - self._super_op_target)
                                +
                                (super_op - self._super_op_target).dag()
                                * (psuper_op_pp[time_ind][param_ind] - self._super_op_target)
                            ).tr().real
                        return self._factor * pl_pp / 2

    class _InterResults(ParameterizedLindbladChannel._InterResults):
        def refresh(self, learning_channel):
            super().refresh(learning_channel)
            if learning_channel._controller.is_time_invariant_for_all_params():
                times_list_len = learning_channel._controller.get_length_times_list()
                ini_params = learning_channel._controller.get_first_parameters(
                    make_copy=False)
                self.C_list_on_times = times_list_len * [
                    learning_channel._model.collapse_op_model(ini_params)]
            else:
                self.C_list_on_times = list(map(
                    learning_channel._model.collapse_op_model,
                    learning_channel._controller.get_parameters(
                        make_copy=False)
                ))

    class _PartialDerivatives(object):
        def __init__(self, learning_channel, *, delay_inter_solve=False):
            if delay_inter_solve:
                self.is_uptodate = False
            else:
                self.refresh(learning_channel)

        def refresh(self, learning_channel, *, check_uptodate=True):
            if check_uptodate and not learning_channel._inter_results.is_uptodate:
                learning_channel._inter_results.refresh(learning_channel)
            parameters_list = learning_channel._controller.get_parameters(
                make_copy=False)
            self.pH_pp_on_times = list(map(
                learning_channel._subroutines.pH_pp,
                parameters_list
            ))
            self.pC_pp_list_on_times = list(map(
                learning_channel._subroutines.pC_pp_list,
                parameters_list
            ))
            if (
                learning_channel._subroutines.pl_pp_channel_part
                is not NO_VALUE
            ) or (
                learning_channel._subroutines.pl_pp_joint_part
                is not NO_VALUE
            ):
                self.pLiou_pp_on_times = [
                    [
                        self.dLiou_dH(pH_ppi) + sum(
                            self.dLiou_dC(Cj, pCj_ppi)
                            for Cj, pCj_ppi in zip(C_list, pC_ppi_list)
                        )
                        for pH_ppi, pC_ppi_list in zip(
                            pH_pp,
                            pC_pp_list
                        )
                    ]
                    for C_list, pH_pp, pC_pp_list in zip(
                        learning_channel._inter_results.C_list_on_times,
                        self.pH_pp_on_times,
                        self.pC_pp_list_on_times,
                    )
                ]
            self.is_uptodate = True

        @staticmethod
        def dLiou_dH(dH):
            return (-1j) * (qt.spre(dH) - qt.spost(dH))

        @staticmethod
        def dLiou_dC(C, dC):
            return qt.sprepost(dC, C.dag()) + qt.sprepost(C, dC.dag()) - (
                qt.spre(dC.dag()*C) + qt.spre(C.dag()*dC)
                + qt.spost(dC.dag()*C) + qt.spost(C.dag()*dC)
            ) / 2

    def __init__(self,
                 model,
                 controller,
                 learning_subroutines,
                 *,
                 make_copy=True,
                 delay_inter_solve=False
                 ):
        super().__init__(
            model=model,
            controller=controller,
            make_copy=make_copy,
            delay_inter_solve=True
        )
        self._inter_results = self._InterResults(
            self, delay_inter_solve=delay_inter_solve)
        self._subroutines = deepcopy(
            learning_subroutines) if make_copy else learning_subroutines
        self._derivatives = self._PartialDerivatives(
            self, delay_inter_solve=delay_inter_solve)

    def update(self,
               input_dm, element_for_calculate_desired_output, learning_rate,
               *, supervisor=NO_VALUE, delay_inter_solve=False, _check_uptodate=True):
        if _check_uptodate:
            if not self._inter_results.is_uptodate:
                self._inter_results.refresh(self)
                self._derivatives.refresh(self, check_uptodate=False)
            elif not self._derivatives.is_uptodate:
                self._derivatives.refresh(self, check_uptodate=False)
        grad_params = self._gradient(
            input_dm, element_for_calculate_desired_output)
        if supervisor is not NO_VALUE:
            grad_params = supervisor._gradient_inspector(grad_params)
        self.set_new_parameters(
            self._controller.get_parameters(make_copy=False)
            - learning_rate * grad_params,
            delay_inter_solve=delay_inter_solve
        )

    def _gradient(self, input_dm, element_for_calculate_desired_output):
        """
        Notes:
            Make sure that self._inter_results and self._derivatives are uptodate before calling this method.
        """
        grad = 0
        if self._subroutines.dl_df is not NO_VALUE:
            grad += self._pl_pp_state_wise_part(
                input_dm, element_for_calculate_desired_output
            )
        if self._subroutines.pl_pp_channel_part is not NO_VALUE:
            grad += self._subroutines.pl_pp_channel_part(self)
        if self._subroutines.pl_pp_joint_part is not NO_VALUE:
            grad += self._subroutines.pl_pp_joint_part(
                self, input_dm, element_for_calculate_desired_output)
        return grad

    def _pl_pp_state_wise_part(self, input_dm, element_for_calculate_desired_output):
        """
        Notes:
            Make sure that self._inter_results and self._derivatives are uptodate before calling this method.
        """
        sup_to_fin = self._inter_results.super_op_to_from_array[-1]
        rho_traj = [
            sup_on_op(super_op, input_dm)
            for super_op in self._inter_results.super_op_from_ini_list
        ]
        C_list_on_times = self._inter_results.C_list_on_times

        pH_pp_on_times = self._derivatives.pH_pp_on_times
        pC_pp_list_on_times = self._derivatives.pC_pp_list_on_times

        len_times_list = self._controller.get_length_times_list()
        # partial loss partial pamameter __ time index _ parameter index
        pl_pp = np.zeros((
            len_times_list,
            self._controller.get_parameter_length()
        ))
        for time_ind, param_ind in it.product(
            range(len_times_list),
            self._controller.get_adjustable_indice(
                make_copy=False
            )
        ):  # TO DO: to optimize (replace the iteration with matrix caculation)
            pl_pp[time_ind, param_ind] = \
                self._subroutines.dl_df(
                    rho_traj[-1],
                    element_for_calculate_desired_output,
                    sup_on_op(
                        sup_to_fin[time_ind],
                        self._f_hat(
                            rho_traj[time_ind],
                            pH_pp_on_times[time_ind][param_ind],
                            pC_pp_list_on_times[time_ind][param_ind],
                            C_list_on_times[time_ind]
                        )
                    )
            )
        return pl_pp

    def set_new_controller(self, new_controller, *, make_copy=True, delay_inter_solve=False):
        self._controller = deepcopy(
            new_controller) if make_copy else new_controller
        if delay_inter_solve:
            self._inter_results.is_uptodate = False
            self._derivatives.is_uptodate = False
        else:
            self._inter_results.refresh(self)
            self._derivatives.refresh(self, check_uptodate=False)

    def set_new_parameters(self, new_parameters_list, *, make_copy=True, delay_inter_solve=False):
        self._controller.set_parameters(
            new_parameters_list, make_copy=make_copy)
        if delay_inter_solve:
            self._inter_results.is_uptodate = False
            self._derivatives.is_uptodate = False
        else:
            self._inter_results.refresh(self)
            self._derivatives.refresh(self, check_uptodate=False)

    def refresh_derivatives(self):
        if not self._derivatives.is_uptodate:
            self._derivatives.refresh(self)

    def refresh_all(self):
        if not self._inter_results.is_uptodate:
            self._inter_results.refresh(self)
        if not self._derivatives.is_uptodate:
            self._derivatives.refresh(self, check_uptodate=False)

    @staticmethod
    def _f_hat(rho, o_H, o_Cs, Cs):
        return -1j*qt.commutator(o_H, rho) \
            + sum(
                o_C*rho*C.dag() + C*rho*o_C.dag() - 0.5*qt.commutator(
                    C.dag()*o_C + o_C.dag()*C,
                    rho,
                    kind='anti'
                )
                for o_C, C in zip(o_Cs, Cs)
        )









