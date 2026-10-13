"""论文后处理：导出 PDF（依赖 pywin32 + 本机 Word）。

在 06_paper.py 生成 docx 之后执行。
"""
import os
import win32com.client

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DOCX = os.path.join(ROOT, "paper", "中国信用利差驱动因素的非线性机制与样本外预测.docx")
PDF = os.path.join(ROOT, "paper", "论文预览.pdf")

word = win32com.client.Dispatch("Word.Application")
word.Visible = False
doc = word.Documents.Open(DOCX, ReadOnly=False)

doc.Save()
doc.ExportAsFixedFormat(PDF, 17)
pages = doc.ComputeStatistics(2)   # wdStatisticPages
words = doc.ComputeStatistics(0)   # wdStatisticWords
print(f"页数: {pages}, 字数: {words}")
doc.Close(False)
word.Quit()
