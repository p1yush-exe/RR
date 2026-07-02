import zipfile
import xml.etree.ElementTree as ET
import sys

def read_docx(path):
    try:
        with zipfile.ZipFile(path) as docx:
            content = docx.read('word/document.xml')
            tree = ET.fromstring(content)
            # The namespace for w:t (text)
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            texts = [node.text for node in tree.findall('.//w:t', ns) if node.text]
            return "".join(texts)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    with open('output.txt', 'w', encoding='utf-8') as f:
        for path in sys.argv[1:]:
            f.write(f"--- {path} ---\n")
            f.write(read_docx(path) + "\n\n")
