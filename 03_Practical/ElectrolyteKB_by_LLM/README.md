#  ElectrolyteKB  
**从文献到设计：锂金属电池电解液的智能挖掘与构效关系建模**

> **Automatically extract electrolyte formulations from scientific literature → Build a structured knowledge base → Predict performance → Recommend optimal recipes.**

---

##  项目简介

锂金属电池（LMB）是下一代高能量密度储能技术的关键，而**电解液设计**直接决定其循环稳定性、库仑效率与安全性。然而，数万篇相关论文中的电解液知识分散在非结构化文本中，难以高效利用。

**ElectrolyteAI** 是一个端到端的智能系统，利用 **大语言模型（LLM）** 自动从海量文献中提取电解液配方、测试条件与性能数据，构建开源数据库 **ElectrolyteKB**，并训练 **机器学习/深度学习模型** 建立“组成-性能”构效关系，最终为**按性能目标反向推荐电解液配方**打下基础。

本项目适用于 AI for Science、材料信息学、电池研发等方向的科研训练或课程设计。

---

##  核心功能

- ✅ **LLM 驱动的文献信息抽取**：从 PDF 论文中自动提取结构化电解液数据  
- ✅ **ElectrolyteKB 数据库**：首个开源的锂金属电池电解液知识库（持续更新）  
- ✅ **构效关系建模**：预测库仑效率（CE）、循环寿命、界面阻抗等关键性能  
- ✅ **智能推荐系统**：输入性能目标（如 CE > 99.5%），输出 Top 5 电解液配方 (附加功能) 
- ✅ **可复现、可扩展**：模块化设计，支持替换 LLM 或 ML 模型 (附加功能)

---

##  推荐参考文献
1. **Dagdelen, J., Dunn, A., Lee, S., et al.**  
   Structured information extraction from scientific text with large language models.  
   *Nature Communications* **2024**, *15*(1), 1418.  
   https://doi.org/10.1038/s41467-024-45674-3  
   > ✅ **核心参考**：展示 LLM 如何从材料科学论文中高精度提取结构化数据（如成分、性能），为本项目提供方法论基础。



