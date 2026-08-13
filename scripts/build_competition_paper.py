from __future__ import annotations

import html
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "competition"
PDF = OUT / "QSNN跨技术路线量子分类研究.pdf"
MD = OUT / "QSNN跨技术路线量子分类研究.md"
METRICS = OUT / "evidence" / "cross_route_metrics.json"
FIG = OUT / "figures"


PAGES = [
    {
        "title": "QSNN 跨技术路线量子分类研究",
        "subtitle": "基于 Kaiwu 相干光 Ising 求解与 WuYue 通用门模型的可复现实现",
        "paragraphs": [
            "摘要：本研究围绕量子随机神经网络的可部署性，建立相干光和通用量子计算机两条互补路线。相干光路线将输入钳制、固定随机储备池、监督读出和双类别吸引子编译为 1000 自旋 Ising 模型，并在 Qboson-1000 真机完成求解；通用量子路线使用 WuYue SDK 构建可训练变分量子分类器，完成量子门操作、全振幅本地验证和云模拟任务。",
            "研究同时设置逻辑回归、RBF-SVM 与模拟退火基线。实验表明，圆环任务在显式加入半径特征后可被经典方法轻易解决，因而本文不主张量子优势。主要贡献是形成从建模、门禁、云提交、结果解码到跨路线审计的完整工程链路，并清晰区分 Lindblad QSNN、CIM 吸引子和幺正门模型。",
            "关键词：量子随机神经网络；相干伊辛机；变分量子分类器；储备池计算；QUBO；WuYue SDK；Kaiwu SDK",
            "版本日期：2026-08-14",
        ],
    },
    {
        "title": "1 问题背景与研究目标",
        "paragraphs": [
            "量子随机神经网络以开放量子随机游走为计算骨架：Hamiltonian 负责相干干涉与特征混合，Lindblad 跳跃负责将概率不可逆地输送至输出节点。该结构在函数逼近、二维分类和文本识别中具有清晰物理含义，但直接在现有硬件上实现任意 Hamiltonian 与任意可训练耗散通道仍然困难。",
            "竞赛要求相干光量子计算机作为核心求解路线，并允许使用通用量子计算机进行协同验证。由此，本研究不强行把同一个数学对象原样搬到两类硬件，而是建立功能一致、物理边界明确的双路线：相干光设备负责大规模 Ising 能量网络，超导门模型负责可编程量子门、线路模拟和变分算法。",
            "研究问题包括：如何将 QSNN 的输入、传播、读出和吸引子编译为 8 位 Ising 矩阵；如何在 Baihua 可接受拓扑上构建完整 VQC；两条量子路线与经典算法相比能说明什么；哪些结果是真机证据，哪些只是模拟或工程近似。",
            "本文的评价原则是可复现优先于名义规模，真实状态优先于预期结果，跨路线比较必须注明样本覆盖、硬件预算和物理语义差异。",
        ],
    },
    {
        "title": "2 QSNN 与开放系统理论",
        "paragraphs": [
            "原始 QSNN 在 N 维希尔伯特空间中使用密度矩阵 rho 描述状态。开放系统演化满足 GKLS 方程：d rho/dt = -i[H,rho] + sum_k(L_k rho L_k^dagger - 1/2{L_k^dagger L_k,rho})。该映射对密度矩阵是线性的，并保持完全正与迹。",
            "仓库 QSNN2D 使用 N_in 个输入基态和两个输出基态。第一阶段在输入子空间执行 exp(-i H_u T_u)；第二阶段使用 L_(o,j)=gamma_(o,j)|o><j| 将输入节点 j 的占据概率定向送入类别输出 o。若忽略相干项，输入概率按 exp(-d_j t) 衰减，输出概率获得不可逆增益。",
            "严格 Lindblad 耗散与本文相干光路线不可混同。Qboson 接收 Ising 耦合矩阵，而不是 H、L_k 和 rho(0)。因此相干光模型只保留传播与吸引子的计算功能；WuYue VQC 则保留可控幺正门和测量，但没有实现可调 Lindblad 跳跃。",
            "这一区分决定了论文措辞：相干光版本称为 QSNN-inspired Ising surrogate；通用量子版本称为 VQC 或 QSNN coherent baseline；只有密度矩阵数值模型才称为严格结构化 Lindblad QSNN。",
        ],
    },
    {
        "title": "3 任务定义、数据与评价指标",
        "paragraphs": [
            "验证任务是二维内外圆环二分类。类别 0 的基础半径为 0.35，类别 1 为 0.80，并叠加标准差 0.06 的径向高斯噪声。角度均匀采样，随机种子固定为 23。VQC 与经典基线使用 144 个样本，其中 96 个训练、48 个测试。",
            "原始数据文件保存 sample_id、x、y、label 和 split；处理后数据增加 radius_squared=x^2+y^2 与 xy。相干光模型使用相同生成规律，但采用 400 个样本的 320/80 划分，以满足既有 1000 自旋实验。",
            "分类指标包括准确率、二元交叉熵和代表样本概率。Ising 求解额外报告物理 Hamiltonian、候选解数量和输入保持率。工程指标包括训练墙钟时间、云任务 ID、设备状态、shots、矩阵规模、耦合数量与系数精度。",
            "圆环标签几乎由半径决定，因此 radius_squared 是强先验。本文将仅使用 x,y 的逻辑回归作为困难基线，并以加入半径后的逻辑回归和 RBF-SVM 暴露任务的真实经典难度。",
        ],
        "image": "dataset_distribution.png",
    },
    {
        "title": "4 总体求解流程",
        "paragraphs": [
            "相干光流程为：生成数据与 128 维温度计编码；固定三层随机储备池；岭回归训练读出；构造样本条件 1000 自旋矩阵；8 位量化；Kaiwu SA 门禁；Qboson CIM 真机；相对自旋解码。",
            "通用量子流程为：生成共同数据；角度编码 x、y、r^2、xy；在 2-3-4 拓扑上构建 RY/RZ/CX 线路；经典优化 8 个参数；WuYue Backend 精确复核；全振幅云模拟；门禁通过后提交 Baihua。",
            "经典流程为：相同训练/测试划分；分别拟合原始线性、增强线性和 RBF 核模型；记录准确率、交叉熵和墙钟时间。所有路径最终写入 JSON/CSV，再由自动脚本生成图表和论文。",
            "真机任务只在本地或云模拟通过后创建。该门禁减少符号、比特序、量化和拓扑错误造成的云额度浪费。",
        ],
        "image": "cross_route_workflow.png",
    },
    {
        "title": "5 QUBO/Ising 建模方法",
        "paragraphs": [
            "QUBO 使用二进制变量 x_i in {0,1} 最小化 x^T Q x；Ising 使用 s_i in {-1,+1} 最小化 -sum J_ij s_i s_j - sum h_i s_i。通过 s_i=2x_i-1 可在二者之间转换，常数项不影响最优解。",
            "本文直接构造 Ising 矩阵，因为 Kaiwu CIMOptimizer 的核心输入就是耦合矩阵。局域场通过偏置自旋 s_b 齐次化：h_i s_i 转为 h_i s_i s_b。于是完整模型只含二体项并保持整体翻转对称性。",
            "样本条件能量由六部分组成：输入钳制 H_in、随机传播 H_prop、监督语义 H_sem、类内铁磁 H_sink、类别互斥 H_mutex 和选择路由 H_route。对每个样本只改变输入场，共享储备池与读出参数。",
            "浮点矩阵按最大绝对值缩放至 [-127,127] 并四舍五入。量化会改变弱边甚至删除极小边，因此量化后的矩阵才是实际硬件模型。代码保存 int16 容器，但有效精度严格为 signed int8。",
        ],
        "formula": "H_total = H_in + H_prop + H_sem + H_sink + H_mutex + H_route",
    },
    {
        "title": "6 1000 自旋相干光模型结构",
        "paragraphs": [
            "模型由 128 个输入自旋、三层各 256 个储备池自旋、两组各 51 个类别吸引子、1 个类别选择自旋和 1 个偏置自旋组成，总计 1000。输入分为 x、y、半径和 xy 四个 32 位温度计银行。",
            "第一层每个节点随机连接 16 个输入节点，后两层每个节点随机连接前层 12 个节点。非零权重取正负归一化常数，随机种子固定，因此储备池是可复现的固定随机特征网络。",
            "矩阵共有 10,644 条独立非零耦合，稀疏度约 2.13%。输入钳制强度为 15，传播系数为 0.24；两类吸引子使用环边 0.70、跨 7 位边 0.25、互斥边 0.20 和弱路由 0.03。",
            "该规模使用 1000 个物理 Ising 模式，不是 log2(1000) 比特的指数空间编码。相干光和门模型的资源口径必须分开陈述。",
        ],
        "table": [
            ["模块", "数量", "索引"], ["输入", "128", "0-127"], ["储备池", "768", "128-895"],
            ["双吸引子", "102", "896-997"], ["选择/偏置", "2", "998-999"],
        ],
    },
    {
        "title": "7 储备池计算与训练方法",
        "paragraphs": [
            "储备池计算使用固定高维动力系统生成特征，只训练简单读出。当前模型不是时间递归 echo-state network，而是三层前馈随机符号储备池：h_l=sign(W_l h_(l-1))。因此它不具备已证明的时序记忆容量。",
            "随机稀疏投影相当于大量随机超平面测试，多层 sign 将输入空间切成离散区域。多尺度语义向量拼接原始 128 维输入与三层 768 维状态，共 896 维。",
            "监督标签映射为 -1/+1，读出 beta 通过岭回归 beta=Z^T(ZZ^T+lambda I)^(-1)t 求得，并做 L1 归一化以控制总耦合预算。当前唯一由标签训练的参数是 beta；储备池位置、符号和强度全部冻结。",
            "经典代理测试达到 100%，但半径输入银行直接提供强判别特征。后续必须删除半径银行、改变随机种子并训练非零储备池权重，才能判断储备池本身的贡献。",
        ],
    },
    {
        "title": "8 双类别吸引子与 CIM 耗散",
        "paragraphs": [
            "每类 51 个自旋通过铁磁环边和长程边形成高磁化低能盆地。896 个语义自旋按模 51 映射到吸引子位置，并以相反符号作用于两个类别群。对应位置的反铁磁边使两类互斥。",
            "CIM 的计算过程包含泵浦、光学损耗、非线性增益饱和、耦合反馈和噪声。连续光场振幅在阈值附近发生分岔，其正负相位表示 Ising 自旋。开放耗散使多个初态可能进入相同稳定盆，因此适合能量吸引子求解。",
            "这种吸引子不是严格 Lindblad 暗态。代码没有构造满足 L_k|psi>=0 的跳跃算符，也没有验证 Liouvillian 谱。正确结论是 CIM 以自身开放动力学求解人为设计的 Ising 能量景观。",
            "候选解频率同样不是自动校准的 Born 概率。程序使用选择自旋与偏置自旋的相对方向进行多数表决，并同时报告候选能量。",
        ],
        "formula": "class = sign(s_selector * s_bias);  H(s) = H(-s)",
    },
    {
        "title": "9 Kaiwu 求解、门禁与真机结果",
        "paragraphs": [
            "程序首先使用 Kaiwu SimulatedAnnealingOptimizer 求解量化后的同一矩阵。门禁要求 8 个测试样本准确率不低于 75%，平均输入保持率不低于 93%。实测为 87.5% 与 93.226%，因而允许创建 CIM 任务。",
            "Qboson-1000 任务 2608130MSS0TJIC01AZ4XHUMYTD0QZZL 返回 10 个候选解。条件样本 320 的真实标签为 0，所有候选均预测 0。CIM 最佳物理能量为 -39794，同一样本本次 SA 最佳值为 -37098。",
            "单次 CIM 能量更低说明该次硬件求解找到了更优候选，但 SA 配置、运行时间和重复预算未严格匹配，不能推出算法优越性。SA 在样本 327 上误分类，也说明当前能量裕量仍不稳定。",
            "完整真机证据保存任务 ID、设备 ID、候选数量、预测和能量，不保存 AK/SK。复现前仍应重新执行 SA 门禁，以防代码或矩阵变化。",
        ],
        "image": "sa_cim_energy.png",
    },
    {
        "title": "10 WuYue 通用量子算法定义",
        "paragraphs": [
            "通用量子算法不是简单调用一个通用门，而是用可组合门集构造具有明确输入、变换、测量和优化目标的完整程序。本文 VQC 包含数据编码 U_enc(x)、参数化变换 U_var(theta)、可观测量测量和经典优化闭环。",
            "线路使用 WuYue SDK 的 QuantumRegister、ClassicalRegister、QuantumCircuit、RY、RZ、CX 与 MEASURE，不依赖 Qiskit 生成线路。9 比特容器适配 Baihua，实际计算位为 q2、q3、q4。",
            "RY 编码 x、y 和 r^2，RZ 编码 xy；2-3 边完成特征纠缠，3-4 边通过 CX-RZ-CX 实现可训练 ZZ 相互作用。最终测量 q4，激发概率作为类别 1 概率。",
            "RY、RZ 和 CX 可组成通用门集意义下的任意幺正近似基础；更重要的是，本线路确实形成了可训练量子分类算法，而不是孤立门操作示例。",
        ],
        "formula": "p1(x;theta) = <psi(x;theta)|(I-Z_readout)/2|psi(x;theta)>",
    },
    {
        "title": "11 VQC 训练算法与代码实现",
        "paragraphs": [
            "VQC 共有 8 个参数。目标函数是训练集平均二元交叉熵。优化器采用 Nelder-Mead，在无解析梯度要求下对小参数模型稳定搜索；初始值依据内外环预期半径给出径向 warm start。",
            "为降低反复构建 SDK 对象的训练开销，代码实现了与 WuYue 线路严格同序的快速状态向量模拟器。训练结束后抽取 6 个样本逐一调用 WuYue Backend.get_device('Full amplitude') 复核，最大概率误差为 3.33e-16。",
            "训练使用 141 次迭代、248 次目标函数评估，墙钟时间 4.523 秒；损失从 0.09616 降至 0.04899。训练与测试准确率均为 100%。",
            "关键代码位于 hardware/wuyue_vqc/wuyue_vqc.py，测试覆盖数据确定性、SDK 一致性、QASM 门集、Baihua 边和读出比特序。",
        ],
        "code": [
            "circuit.add(RY, qreg[q0], paras=pi/2*(x+1))",
            "circuit.add(CX, qreg[q1], control=qreg[q0])",
            "circuit.add(CX, qreg[q2], control=qreg[q1])",
            "circuit.add(RZ, qreg[q2], paras=theta_zz)",
            "circuit.add(CX, qreg[q2], control=qreg[q1])",
        ],
    },
    {
        "title": "12 WuYue 模拟、云提交与 Baihua 状态",
        "paragraphs": [
            "执行顺序严格为本地精确验证、WuYue 全振幅云模拟、Baihua 真机。只有 SDK 概率一致且测试准确率不低于 80% 时，程序才允许提交真机。",
            "云模拟任务 2608140MSS8C2UB01APUH2W4T1C3B1SW 在 WuYue-QPUSim-FullAmpSim 上计算成功，返回 1024 shots。代表测试样本标签为 1，精确 p1=0.96271，云采样 p1=0.972656，预测一致。",
            "2026-08-14 提交同一线路到 WuYue-QPU-Baihua 时，服务端返回‘维护中’，任务未创建。这是外部设备状态阻塞，不是本地线路验证失败。论文不能将其记为真机成功。",
            "此前固定参数 9 比特相干线路任务曾获平台接受，但长时间保持状态码 1。它只能证明接口和拓扑曾被接受，不能替代本次可训练 VQC 的真机结果。设备恢复后可用同一命令直接补测。",
        ],
        "table": [
            ["阶段", "状态", "证据"], ["WuYue 本地", "通过", "误差 3.33e-16"],
            ["全振幅云模拟", "成功", "1024 shots"], ["Baihua", "维护中", "未创建任务"],
        ],
    },
    {
        "title": "13 求解结果汇总",
        "paragraphs": [
            "通用量子路线在共同 48 个测试样本的精确模拟上达到 100%。云模拟只对代表样本执行 1024 shots，预测正确。相干光路线的经典储备池代理为 100%，SA 抽查为 87.5%，CIM 真机代表样本预测正确。",
            "这些数字的覆盖范围不同：VQC 100% 是全测试集模拟结果，CIM 结果是单样本真机结果，SA 87.5% 只来自 8 个样本。任何不标注分母的横向柱状图都会产生误导。",
            "相干光的核心证据是 1000 自旋、10,644 条耦合和真实 CIM 返回；VQC 的核心证据是完整可训练通用门算法和 WuYue 云模拟；经典基线用于说明任务难度。",
            "Baihua 当前没有完成结果，因此总体验收状态应写为：相干光真机完成，通用量子云模拟完成，超导真机待设备恢复补测。",
        ],
        "image": "route_accuracy_comparison.png",
    },
    {
        "title": "14 经典算法对比",
        "paragraphs": [
            "仅用 x,y 的逻辑回归测试准确率为 50%，因为同心圆不能被单条线性边界分离。加入 r^2 和 xy 后，逻辑回归达到 100%；RBF-SVM 在原始 x,y 上也达到 100%。",
            "单次墙钟测量中，两种成功经典模型训练约 0.002 秒，VQC 训练约 4.523 秒。计时环境和实现语言不同，不能作为严格加速基准，但足以说明当前小数据任务没有量子速度优势。",
            "VQC 的意义在于完成通用门模型协同验证；相干光的意义在于展示大规模 Ising 物理求解链路。若将 100% 准确率作为创新点，会被经典基线立即否定。",
            "更有区分度的后续任务应删除显式半径特征，增加分布外半径、角向结构、随机标签、噪声扰动和多随机种子，并在相同样本与调用预算下统计置信区间。",
        ],
        "table": [
            ["方法", "特征", "测试准确率"], ["逻辑回归", "x,y", "50%"],
            ["逻辑回归", "x,y,r2,xy", "100%"], ["RBF-SVM", "x,y", "100%"],
            ["WuYue VQC", "角度编码", "100%（模拟）"],
        ],
    },
    {
        "title": "15 跨技术路线物理分析",
        "paragraphs": [
            "相干光 CIM 的有效计算是开放、受驱动、耗散且非幺正的。固定损耗和增益饱和不是独立可训练 Lindblad 参数，但它们能与可编程 Ising 耦合共同形成稳定吸引盆。",
            "超导门模型主要执行可逆幺正变换。设备固有 T1/T2 退相干通常是非任务定向误差，不能自动实现类别相关的 L_(o,j)。没有动态 reset 或辅助系统丢弃时，本文只实现 VQC，而不声称耗散 QSNN。",
            "因此，相干光更适合当前 1000 节点能量网络和吸引子求解；超导更适合验证可编程量子门、纠缠、线路模拟和通用算法。二者不是互相替代，而是分别验证 QSNN 架构中的能量吸引与相干计算侧面。",
            "严格跨路线科学比较需要定义共同输入输出接口与共同成本指标，而不是把 Ising 自旋数直接等同于超导 qubit 数。",
        ],
    },
    {
        "title": "16 求解时长与资源核算",
        "paragraphs": [
            "VQC 本地优化用时 4.523 秒，训练集精确评估约 0.018 秒，测试集评估为毫秒量级。WuYue 云模拟任务返回成功，但当前 SDK 报告未提供服务端精确计算时长，因此本文不虚构云端时延。",
            "相干光报告记录 SA 与 CIM 的能量和任务 ID，但未保存提交、排队、计算和下载的分段时间。论文只报告流程和任务状态，不以缺失的墙钟时间构造加速比。",
            "经典逻辑回归与 RBF-SVM 单次训练约 0.002 至 0.015 秒。该数字受进程启动、CPU 和库版本影响，仅用于数量级说明。",
            "后续应统一记录 created_at、queued_at、started_at、finished_at、客户端下载耗时、shots、候选数和云额度，并以同一停止准则比较 SA、CIM 和经典优化器。",
        ],
        "table": [
            ["项目", "规模", "已记录时长"], ["VQC 训练", "96 样本/8 参数", "4.523 s"],
            ["经典训练", "96 样本", "0.002-0.015 s"], ["WuYue 云模拟", "9 qubit/1024 shots", "平台未返回分段时长"],
            ["CIM", "1000 spins/10 solutions", "平台未返回分段时长"],
        ],
    },
    {
        "title": "17 模型亮点与创新点",
        "paragraphs": [
            "第一，模型把 1000 个自旋全部分配为有语义的输入、储备池、吸引子、选择和偏置模块，而不是用随机满矩阵填充设备容量。矩阵经过真实 8 位量化和 SA 门禁后才提交真机。",
            "第二，建立了严格的概念分层：Lindblad QSNN、CIM 耗散吸引子、固定随机储备池和门模型 VQC 分别定义，避免以‘耗散’一词掩盖物理差异。",
            "第三，通用量子协同路线不是门操作 smoke test，而是包含数据编码、纠缠、可训练参数、损失函数、经典优化、SDK 精确核验和云模拟的完整算法。",
            "第四，所有关键结论由 JSON、CSV、矩阵、参数文件、任务 ID 和自动图表支撑；失败状态也被保留。项目的创新点是跨技术路线的可执行建模与证据链，而非未经证实的量子优势。",
        ],
    },
    {
        "title": "18 局限性、风险与改进",
        "paragraphs": [
            "圆环任务具有显式径向捷径，数据规模较小。VQC 和相干光输入都利用半径相关特征，因此高准确率不能代表普适量子学习能力。",
            "相干光储备池参数完全冻结，只训练岭回归读出；CIM 只运行一个条件样本。下一步需要量化感知训练储备池非零权重，并扩展完整测试集和多次真机重复。",
            "VQC 的有效计算只使用 3 个 qubit，9 qubit 主要是 Baihua 拓扑容器；Baihua 本轮因维护未执行。必须在设备恢复后完成代表样本组，而不是用云模拟冒充真机。",
            "当前没有统一功耗、云额度、排队时间和调参预算，也没有统计显著性检验。所有优势表述应限制为‘链路完成’、‘候选能量更低’或‘模拟准确率达到’，不能上升为速度或精度优势。",
        ],
    },
    {
        "title": "19 结论、复现与参考资料",
        "paragraphs": [
            "本研究完成了两条可执行路线。Kaiwu 相干光模型在 Qboson-1000 上完成 1000 自旋真机求解；WuYue 通用量子模型完成可训练 VQC、本地全振幅核验和云模拟。Baihua 真机提交因平台维护受阻，状态已明确记录。",
            "实验未显示量子优势：带半径特征的逻辑回归和 RBF-SVM 均达到 100%。研究价值在于模型边界清楚、代码与数据齐全、真机证据可追踪、跨路线差异可审计。",
            "复现入口：hardware/kaiwu_qsnn_photonic/large_scale_qsnn.py；hardware/wuyue_vqc/wuyue_vqc.py；scripts/build_cross_route_evidence.py。运行说明见 docs/competition/支撑材料说明.md。",
            "参考资料：Wang et al., Implementation of quantum stochastic walks for function approximation (2022)；WuYueSDK 官方开源仓库 https://gitee.com/WUYUEQbit/WuYueSDK；Kaiwu-PyTorch-Plugin https://github.com/qboson/kaiwu-pytorch-plugin；项目内相干光完整理论报告。",
            "最终结论：当前最成熟的核心求解是相干光 Ising 真机路线；通用门模型已经满足算法构建和云模拟要求，但超导真机结果仍待 Baihua 恢复后补齐。",
        ],
    },
]


def markdown() -> str:
    lines = []
    for index, page in enumerate(PAGES):
        level = "#" if index == 0 else "##"
        lines.append(f"{level} {page['title']}")
        if page.get("subtitle"):
            lines.extend(("", f"**{page['subtitle']}**"))
        for paragraph in page.get("paragraphs", []):
            lines.extend(("", paragraph))
        if page.get("formula"):
            lines.extend(("", "```text", page["formula"], "```"))
        if page.get("code"):
            lines.extend(("", "```python", *page["code"], "```"))
        if page.get("table"):
            table = page["table"]
            lines.extend(("", "| " + " | ".join(table[0]) + " |", "|" + "|".join("---" for _ in table[0]) + "|"))
            lines.extend("| " + " | ".join(row) + " |" for row in table[1:])
        if page.get("image"):
            lines.extend(("", f"![{page['title']}](figures/{page['image']})"))
        if index != len(PAGES) - 1:
            lines.extend(("", "---"))
    return "\n".join(lines) + "\n"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(2.0 * cm, 1.0 * cm, "QSNN cross-technology quantum classification study")
    canvas.drawRightString(A4[0] - 2.0 * cm, 1.0 * cm, f"{doc.page}")
    canvas.restoreState()


def build_pdf():
    font_candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    )
    font_path = next((path for path in font_candidates if path.exists()), None)
    if font_path is None:
        raise FileNotFoundError("No supported CJK font found; install Noto Sans CJK.")
    pdfmetrics.registerFont(TTFont("CJKBody", str(font_path)))
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ChineseTitle", parent=styles["Title"], fontName="CJKBody", fontSize=22, leading=30, textColor=colors.HexColor("#17324D"), alignment=TA_CENTER, spaceAfter=16)
    subtitle = ParagraphStyle("ChineseSubtitle", parent=styles["Normal"], fontName="CJKBody", fontSize=12, leading=19, textColor=colors.HexColor("#4F6070"), alignment=TA_CENTER, spaceAfter=20)
    heading = ParagraphStyle("ChineseHeading", parent=styles["Heading1"], fontName="CJKBody", fontSize=17, leading=23, textColor=colors.HexColor("#17324D"), spaceAfter=12)
    body = ParagraphStyle("ChineseBody", parent=styles["BodyText"], fontName="CJKBody", fontSize=9.4, leading=15.2, alignment=TA_LEFT, textColor=colors.HexColor("#202B33"), spaceAfter=8)
    formula = ParagraphStyle("Formula", parent=body, fontName="Courier", fontSize=8, leading=12, backColor=colors.HexColor("#F2F5F7"), borderPadding=7, spaceBefore=5, spaceAfter=10)
    doc = SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=1.8 * cm, leftMargin=1.8 * cm, topMargin=1.65 * cm, bottomMargin=1.55 * cm, title=PAGES[0]["title"], author="QSNN Project Team")
    story = []
    for index, page in enumerate(PAGES):
        if index == 0:
            story.extend((Spacer(1, 2.0 * cm), Paragraph(page["title"], title), Paragraph(page["subtitle"], subtitle), Spacer(1, 0.5 * cm)))
        else:
            story.append(Paragraph(page["title"], heading))
        for paragraph in page.get("paragraphs", []):
            story.append(Paragraph(html.escape(paragraph), body))
        if page.get("formula"):
            story.append(Paragraph(html.escape(page["formula"]), formula))
        if page.get("code"):
            story.append(Paragraph("<br/>".join(html.escape(line) for line in page["code"]), formula))
        if page.get("table"):
            data = [[Paragraph(html.escape(cell), body) for cell in row] for row in page["table"]]
            table = Table(data, colWidths=[(A4[0] - 3.6 * cm) / len(data[0])] * len(data[0]), repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "CJKBody"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAB4BD")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F6F8FA")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.extend((Spacer(1, 5), table))
        if page.get("image"):
            path = FIG / page["image"]
            image = Image(str(path))
            image._restrictSize(A4[0] - 4.0 * cm, 7.3 * cm)
            story.extend((Spacer(1, 8), image))
        if index != len(PAGES) - 1:
            story.append(PageBreak())
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    if not METRICS.exists():
        raise FileNotFoundError("Run scripts/build_cross_route_evidence.py first.")
    json.loads(METRICS.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    MD.write_text(markdown(), encoding="utf-8")
    build_pdf()
    print(json.dumps({"markdown": str(MD), "pdf": str(PDF), "planned_pages": len(PAGES)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
