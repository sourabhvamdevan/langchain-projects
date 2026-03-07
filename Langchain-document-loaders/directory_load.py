from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader




loader=DirectoryLoader(
    path='my-folder',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs=loader.load()
print(len(docs))

print(docs[400].page_content)
