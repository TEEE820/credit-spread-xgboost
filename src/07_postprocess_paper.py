"""论文后处理：插入作者简介首页脚注（圈码①），导出 PDF。

依赖 pywin32 + 本机 Word。在 06_paper.py 生成 docx 之后执行。
"""
import os
import win32com.client

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DOCX = os.path.join(ROOT, "paper", "中国信用利差驱动因素的非线性机制与样本外预测.docx")
PDF = os.path.join(ROOT, "paper", "论文预览.pdf")

AUTHOR_LINE = "童欣慧（华南师范大学国际商学院）"
INTRO_TEXT = "作者简介：童欣慧，女，华南师范大学国际商学院本科生。"

word = win32com.client.Dispatch("Word.Application")
word.Visible = False
doc = word.Documents.Open(DOCX, ReadOnly=False)

rng = doc.Content
found = rng.Find.Execute(AUTHOR_LINE)
if not found:
    print("ERROR: 未找到作者行，脚注未插入")
else:
    rng.Collapse(0)  # wdCollapseEnd：定位到作者姓名末尾
    # Reference 传圈码"①"作为自定义引用标记（模版要求①②③编号）
    fn = doc.Footnotes.Add(rng, "①", INTRO_TEXT)
    fn.Range.Font.Name = "Times New Roman"
    fn.Range.Font.NameFarEast = "宋体"
    fn.Range.Font.Size = 9
    print("脚注已插入:", INTRO_TEXT)

doc.Save()
doc.ExportAsFixedFormat(PDF, 17)
pages = doc.ComputeStatistics(2)   # wdStatisticPages
words = doc.ComputeStatistics(0)   # wdStatisticWords
print(f"脚注数量: {doc.Footnotes.Count}, 页数: {pages}, 字数: {words}")
doc.Close(False)
word.Quit()
