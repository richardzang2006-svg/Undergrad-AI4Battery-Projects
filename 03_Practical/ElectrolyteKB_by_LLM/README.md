#  ElectrolyteKB  
**从文献到设计：锂金属电池电解液的智能挖掘与构效关系建模**

> **Automatically extract electrolyte formulations from scientific literature → Build a structured knowledge base → Predict performance → Recommend optimal recipes.**

---

##  项目简介

锂金属电池（LMB）是下一代高能量密度储能技术的关键，而**电解液设计**直接决定其循环稳定性、库仑效率与安全性。然而，数万篇相关论文中的电解液知识分散在非结构化文本中，难以高效利用。

**ElectrolyteAI** 是一个端到端的智能系统(整体目标)，利用 **大语言模型（LLM）** 自动从海量文献中提取电解液配方、测试条件与性能数据，构建开源数据库 **ElectrolyteKB**，并训练 **机器学习/深度学习模型** 建立“组成-性能”构效关系，最终为**按性能目标反向推荐电解液配方**打下基础。

**ElectrolyteAI** 采用 **多智能体协同架构**，通过四个阶段实现从文献到智能设计的闭环：
1. **文献数据提取 Agent**：微调 LLM 自动抽取电解液配方与性能  
2. **知识构建 Agent**：清洗、标准化、构建结构化数据库与知识图谱  
3. **性能预测 模型**：训练 ML/DL 模型建立“组成–性能”构效关系  
4. **交互推荐 Agent**：提供聊天式界面，输入配方即可获得性能预测与优化建议

本项目融合大语言模型、材料信息学与人机交互，打造面向下一代电池研发的 AI 助手，适用于 AI for Science、材料信息学、电池研发等方向的科研训练或课程设计。

---

##  核心功能

- ✅ **LLM 驱动的文献信息抽取**：从 PDF 论文中自动提取结构化电解液数据  
- ✅ **ElectrolyteKB 数据库**：首个开源的锂金属电池电解液知识库（持续更新）  
- ✅ **构效关系建模**：预测库仑效率（CE）、循环寿命、界面阻抗等关键性能  
- ✅ **智能推荐系统**：输入性能目标（如 CE > 99.5%），输出 Top 5 电解液配方 (附加功能) 
- ✅ **可复现、可扩展**：模块化设计，支持替换 LLM 或 ML 模型 (附加功能)

---

##  推荐技术路线

### （1）文献数据提取 Agent：微调 LLM 实现高精度信息抽取
- 收集大量高质量锂金属电池电解液文献
- **人工标注少量样本**（200~500 条），构建微调数据集（JSON 格式）
- 微调开源 LLM（如 Llama-3-8B 或 Qwen2）或使用 LLM API + Prompt Engineering
- 输出结构化字段：溶剂/锂盐/添加剂、浓度、测试条件、性能指标（CE、循环数等）

### （2）知识构建 Agent：构建结构化电解液知识库与知识图谱
- 对抽取数据进行：
  - 单位标准化（如 “1M” → “1.0 mol/L”）
  - 化学名称归一化（“ethylene carbonate” → “EC”）
  - 去重与一致性校验
- 构建 **ElectrolyteKB** 数据库
- 进一步构建 **电解液知识图谱**：
  - 节点：分子（SMILES）、性能指标、文献
  - 边：组成关系、性能关联、引用关系、计算模拟（可选）

### （3）性能预测 Agent：训练构效关系模型
- 特征工程：
  - 分子层面：SMILES → 分子描述符（LogP, MW, HBD/HBA）或图表示（GNN）、元素组成等等
  - 配方层面：浓度、溶剂介电常数加权、Donor Number 总和等
- 模型选择：
  - 回归任务（预测 CE、阻抗）：XGBoost、Random Forest、D-MPNN
- 输出：可解释的性能预测 + 不确定性估计

### （4）交互推荐 Agent：聊天式智能电解液顾问
- 基于 Gradio / Streamlit 构建 Web 聊天界面
- 支持两种交互模式：
  - **模式 A（预测）**：用户输入配方 → 返回预测 CE、循环寿命、稳定性分析  
    > *例：输入 “1M LiTFSI in DME:DOL (1:1) + 2% LiNO₃”*
  - **模式 B（推荐）**：用户输入目标 → 返回 Top 3 推荐配方  
    > *例：目标 “CE > 99%，成本 < $50/L，室温”*
- 后端融合知识图谱（查文献支持） + 预测模型（给性能） + 规则引擎（判成本/可合成性）

---

###  主要参考文献

1. **Dagdelen, J.; Dunn, A.; Lee, S.; et al.**  
   Structured information extraction from scientific text with large language models.  
   *Nat. Commun.* **2024**, *15*(1), 1418.  
   https://doi.org/10.1038/s41467-024-45563-x  
   > 系统验证大语言模型（LLM）从材料科学文献中高精度提取结构化数据（如成分、性能、条件）的工作，为本项目信息抽取模块提供核心方法论。

2. **Kim, S. C.; Oyakhire, S. T.; Athanitis, C.; et al.**  
   Data-driven electrolyte design for lithium metal anodes.  
   *Proc. Natl. Acad. Sci. U.S.A.* **2023**, *120*(10), e2214357120.  
   https://doi.org/10.1073/pnas.2214357120  
   > 基于小规模电解液数据集构建简单机器学习模型，首次建立锂金属负极电解液“组成–库仑效率”构效关系，验证数据驱动设计的可行性。

3. **Robson, M. J.; Xu, S.; Wang, Z.; et al.**  
   Multi-Agent-Network-Based Idea Generator for Zinc-Ion Battery Electrolyte Discovery: A Case Study on Zinc Tetrafluoroborate Hydrate-Based Deep Eutectic Electrolytes.  
   *Adv. Mater.* **2025**, 2502649.  
   https://doi.org/10.1002/adma.202502649  
   > 利用多智能体协作生成新型电解液配方，展示 AI 从“数据挖掘”迈向“创意生成”的前沿范式，启发本项目推荐系统设计。

4. **Hu, X.; Chen, S.; Chen, L.; et al.**  
   Automating structure-activity analysis for electrochemical nitrogen reduction catalyst design through multi-agent collaborations.  
   *Natl. Sci. Rev.* **2025**, nwaf372.  
   https://doi.org/10.1093/nsr/nwaf372  
   >  多智能体协同完成“文献挖掘→特征提取→构效建模→设计建议”全流程，为本项目端到端架构提供系统级参考。

5. **Kang, Y.; Lee, W.; Bae, T.; et al.**  
   Harnessing large language models to collect and analyze metal–organic framework property data set.  
   *J. Am. Chem. Soc.* **2025**, *147*(5), 3943–3958.  
   https://doi.org/10.1021/jacs.4c11085  
   >  展示 LLM 在化学材料领域大规模数据集构建中的可靠性与可扩展性，支持本项目 ElectrolyteKB 的构建策略。



