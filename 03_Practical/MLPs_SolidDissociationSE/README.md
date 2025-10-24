# MLPs for Solid Dissociation Electrolyte Structures and Their Ion Diffusion Mechanisms 
**新型固态电解质结构及其离子扩散机制解析  **

>**Exploring Atomic-Scale Mechanisms of High Ionic Conductivity in Novel Solid-State Electrolytes via Machine Learning Potentials**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![DeepMD](https://img.shields.io/badge/MLP-DeepMD%2FACE%2FMTP-lightgrey)](https://github.com/deepmodeling/deepmd-kit)

> **本科生课外科研项目训练 · AI for Battery Materials**  
> 项目归属：[Undergrad-AI4Battery-Projects](https://github.com/zhangdft/Undergrad-AI4Battery-Projects)

---

## 📌 项目背景

固态电解质因其**不可燃性、低温高离子电导率和环境友好性**，被视为下一代高安全电池的关键材料。然而，传统固态电解质的离子传输性能高度依赖于特定晶体结构，掺杂或组分调控极易导致**离子通道堵塞、相变或副相生成**，严重限制了“电解质工程”的灵活性。

近期，宁波东方理工大学孙学良院士、李晓娜副教授与有研（广东）新材料研究院梁剑文研究员提出了一种创新的 **“固相解离”（Solid Dissociation）** 设计策略：利用范德华晶体 **M(Oₘ)Clₙ**（M = Ta, Nb, Zr, Hf, Al, Y, In 等）作为“固态溶剂”，可高效溶解多种金属盐（从简单阴离子盐到聚阴离子盐），成功构建 **70+ 种新型超离子导体**，其中 **40+ 种在室温下离子电导率 > 10⁻³ S cm⁻¹**。

然而，这类材料的**原子级结构形成机制、离子扩散路径及高电导起源**尚不明确，亟需结合先进计算模拟手段进行深入解析。

---

## 🎯 项目目标

本项目旨在利用**机器学习势（Machine Learning Potentials, MLPs）**，从原子尺度揭示新型固相解离型固态电解质的结构-性能关系与离子输运机制，具体包括：

1. **构建高精度MLP模型**（基于 DeepMD、ACE 或 MTP）用于目标电解质体系的分子动力学模拟；
2. **解析电解质的局部与长程结构特征**（如配位环境、无序度、通道连通性）及其与离子电导率的关联；
3. **揭示锂/钠离子扩散机制**，包括：
   - 扩散路径网络
   - 阴离子骨架的动态作用
   - 离子集体输运行为（correlated motion）
4. **建立结构有序度（如晶格畸变、位点占有率）与离子电导的定量关系**；
5. （可选）提出**高电导电解质的理性设计策略**。

---

## 🔬 研究方法

- **第一性原理计算**（DFT）：生成训练数据，验证关键结构与能量；
- **机器学习势开发**：使用 [DeepMD-kit](https://github.com/deepmodeling/deepmd-kit)、[ACE](https://github.com/ACEsuit/ace) 或 [MTP](https://gitlab.com/ashapeev/mtp) 构建高精度、高效率势函数；
- **大规模分子动力学（MD）模拟**：在纳秒-微秒尺度模拟离子扩散行为；
- **结构与动力学分析**：
  - 径向分布函数（RDF）、配位数分析
  - 均方位移（MSD）→ 离子电导率估算
  - 扩散路径可视化（如 NEB、Dijkstra 算法）
  - 动态结构因子、振动谱分析

---

## 📂 项目结构
