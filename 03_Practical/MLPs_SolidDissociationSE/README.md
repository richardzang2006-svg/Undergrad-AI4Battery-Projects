# MLPs for Solid Dissociation Electrolyte Structures and Their Ion Diffusion Mechanisms 
**新型固态电解质结构及其离子扩散机制解析  **

>**Exploring Atomic-Scale Mechanisms of High Ionic Conductivity in Novel Solid-State Electrolytes via Machine Learning Potentials**

> **本科生课外科研项目训练 · AI for Battery Materials**  
> 项目归属：[Undergrad-AI4Battery-Projects](https://github.com/zhangdft/Undergrad-AI4Battery-Projects)

---

##  项目背景

固态电解质因其**不可燃性、低温高离子电导率和环境友好性**，被视为下一代高安全电池的关键材料。然而，传统固态电解质的离子传输性能高度依赖于特定晶体结构，掺杂或组分调控极易导致**离子通道堵塞、相变或副相生成**，严重限制了“电解质工程”的灵活性。

近期，研究人员提出了一种创新的 **“固相解离”（Solid Dissociation）** 设计策略：利用范德华晶体 **M(Oₘ)Clₙ**（M = Ta, Nb, Zr, Hf, Al, Y, In 等）作为“固态溶剂”，可高效溶解多种金属盐（从简单阴离子盐到聚阴离子盐），成功构建 **70+ 种新型超离子导体**，其中 **40+ 种在室温下离子电导率 > 10⁻³ S cm⁻¹**。

然而，这类材料的**原子级结构形成机制、离子扩散路径及高电导起源**尚不明确，亟需结合先进计算模拟手段进行深入解析。

---

##  项目目标

本项目旨在利用**机器学习势（Machine Learning Potentials, MLPs）**，从原子尺度揭示新型固相解离型固态电解质的结构-性能关系与离子输运机制，具体包括：

1. **构建高精度MLP模型**（基于 DeepMD、ACE 或 MTP）用于目标电解质体系的分子动力学模拟；
2. **解析电解质的局部与长程结构特征**（如配位环境、无序度、通道连通性）及其与离子电导率的关联；
3. **揭示锂离子扩散机制**，包括：
   - 扩散路径网络
   - 阴离子骨架的动态作用
   - 离子集体输运行为（correlated motion）
4. **建立结构有序度与离子电导的定量关系**；
5. （可选）提出**高电导电解质的理性设计策略**。

---

##  研究方法

- **第一性原理计算**（DFT）：生成训练数据，验证关键结构与能量；
- **机器学习势开发**：构建高精度、高效率势函数；
- **大规模分子动力学（MD）模拟**：在纳秒-微秒尺度模拟离子扩散行为；
- **结构与动力学分析**：
  - 径向分布函数（RDF）、配位数分析
  - 均方位移（MSD）→ 离子电导率估算
  - 扩散路径可视化
  - 动态结构因子、振动谱分析

---

##  参考文献

1. **Yue J, Zhang S, Wang X, et al.** Universal superionic conduction via solid dissociation of salts in van der Waals materials. *Nature Energy*, 2025, 1–14. doi:[10.1038/s41560-025-01853-2](https://doi.org/10.1038/s41560-025-01853-2).  
   > 首次提出“固相解离”策略，实现范德华材料中普适性超离子导电。

2. **Jun K J, Chen Y, Wei G, et al.** Diffusion mechanisms of fast lithium-ion conductors. *Nature Reviews Materials*, 2024, 9(12): 887–905. doi:[10.1038/s41578-024-00715-9](https://doi.org/10.1038/s41578-024-00715-9).  
   > 全面综述固态电解质中离子扩散的微观机制、瓶颈与优化路径。

3. **Zhao F, Zhang S, Wang S, et al.** Anion sublattice design enables superionic conductivity in crystalline oxyhalides. *Science*, 2025, 390(6769): 199–204. doi:[10.1126/science.adt9678](https://doi.org/10.1126/science.adt9678).  
   > 揭示阴离子骨架柔性与局域极化对高离子电导的关键作用。

4. **He X, Zhu Y, Mo Y.** Origin of fast ion diffusion in super-ionic conductors. *Nature Communications*, 2017, 8: 15893. doi:[10.1038/ncomms15893](https://doi.org/10.1038/ncomms15893).  
   > 阐明阳离子集体协同输运（correlated migration）是超离子导体中快速扩散的核心机制。

5. **Chen Z, Du T, Krishnan N M A, et al.** Disorder-induced enhancement of lithium-ion transport in solid-state electrolytes. *Nature Communications*, 2025, 16: 1057. doi:[10.1038/s41467-025-56322-x](https://doi.org/10.1038/s41467-025-56322-x).  
   > 证明结构无序对离子扩散的影响。

6. **Singh B, Wang Y, Liu J, et al.** Critical role of framework flexibility and disorder in driving high ionic conductivity in LiNbOCl4. *Journal of the American Chemical Society*, 2024, 146(25): 17158-17169. doi:[10.1021/jacs.4c03142](https://doi.org/10.1038/s41467-025-56322-x).  
   > 证明骨架和结构无序的影响。

