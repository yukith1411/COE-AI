import warnings
warnings.filterwarnings("ignore")

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


legal_text = """
THE CODE ON WAGES, 2019
Chapter 1: Preliminary

Short title, extent and commencement.—(1) This Act may be called the Code on Wages, 2019.
(2) It extends to the whole of India.

Chapter 2: Minimum Wages
6. Fixation of minimum wages.—(1) The appropriate Government shall fix the minimum rate of wages payable to employees employed in an employment specified in the Schedule.
(2) For the purpose of sub-section (1), the appropriate Government shall fix a minimum rate of wages for time work, piece work, or guaranteed time rate.

Power of Central Government to fix floor wage.—(1) The Central Government shall fix floor wage taking into account minimum living standards of a worker in such manner as may be prescribed.

Chapter 3: Payment of Wages
17. Mode of payment of wages.—For the purposes of this Code, all wages shall be paid in current coin or currency notes or by bank transfer to the bank account of the employee or by the electronic mode.
18. Fixation of wage period.—The employer shall fix the wage period for employees either as daily or weekly or fortnightly or monthly.
"""


def create_vector_db():
    print("Processing Legal Data into the Brain...")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    texts = text_splitter.split_text(legal_text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS vector database
    vectorstore = FAISS.from_texts(texts, embeddings)

    print("Vector Database Created Successfully!")

    return vectorstore


if __name__ == "__main__":
    db = create_vector_db()
