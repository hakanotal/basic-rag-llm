"""Streamlit app for RAG system."""

import streamlit as st
import os
from pathlib import Path
from src import settings, EmbeddingGenerator, VectorStore, Retriever, Generator, DocumentProcessor, TextChunker

st.set_page_config(page_title="RAG - Document Q&A", page_icon="📚", layout="wide")
st.title("📚 RAG - Document Q&A System | CIST 533 Final Project")
st.markdown("Ask questions about your uploaded documents!")

# Check for Gemini API key
if not os.getenv('GEMINI_API_KEY'):
    st.error("⚠️ GEMINI_API_KEY environment variable not set!")
    st.info("Set it in Streamlit Cloud: Settings → Secrets → Add: `GEMINI_API_KEY = \"your-key-here\"`")
    st.stop()

def ensure_sample_indexed(vector_store, embedder):
    """Ensure sample PDF is indexed on startup (for Streamlit Cloud)."""
    try:
        info = vector_store.get_collection_info()
        if info['points_count'] == 0:
            # Look for any PDF in uploads directory
            pdf_files = list(settings.uploads_dir.glob("*.pdf"))
            if pdf_files:
                sample_pdf = pdf_files[0]  # Use first PDF found
                st.info(f"🔄 Indexing demo document: {sample_pdf.name}...")
                processor = DocumentProcessor()
                documents = [processor.process_pdf(sample_pdf)]
                chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)
                chunks = chunker.chunk_documents(documents)
                chunks_with_embeddings = embedder.embed_chunks(chunks)
                vector_store.add_chunks(chunks_with_embeddings)
                st.success(f"✅ Indexed {len(chunks)} chunks from {sample_pdf.name}")
    except Exception as e:
        st.warning(f"Could not auto-index sample: {str(e)}")

@st.cache_resource
def load_components():
    try:
        embedder = EmbeddingGenerator(settings.gemini_embedding_model)
        vector_store = VectorStore(settings.collection_name, str(settings.chroma_dir))
        
        # Auto-index sample PDF if database is empty
        ensure_sample_indexed(vector_store, embedder)
        
        retriever = Retriever(embedder, vector_store)
        generator = Generator(settings.gemini_llm_model)
        info = vector_store.get_collection_info()
        return retriever, generator, info, None
    except Exception as e:
        return None, None, None, str(e)

retriever, generator, collection_info, error = load_components()

with st.sidebar:
    
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type=['pdf'], accept_multiple_files=True, help="Upload PDF documents to index")
    
    if uploaded_files:
        if st.button("💾 Save Uploaded Files", type="primary", use_container_width=True):
            success_count = 0
            for uploaded_file in uploaded_files:
                try:
                    file_path = settings.uploads_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    success_count += 1
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            if success_count > 0:
                st.success(f"✓ Saved {success_count} file(s)")
                st.rerun()
    
    st.subheader("Indexed Documents")
    uploaded_pdfs = list(settings.uploads_dir.glob("*.pdf"))
    if uploaded_pdfs:
        for pdf in uploaded_pdfs:
            st.text(f"📄 {pdf.name}")
    else:
        st.info("No PDFs in uploads folder")
    
    # Reindex button
    if uploaded_pdfs:
        if st.button("🔄 Reindex Documents", use_container_width=True):
            with st.spinner("Reindexing documents..."):
                try:
                    # Delete existing collection
                    if not error and collection_info:
                        try:
                            vs = VectorStore(settings.collection_name, str(settings.chroma_dir))
                            vs.delete_collection()
                            st.info("✓ Cleared existing index")
                        except:
                            pass
                    
                    # Process PDFs
                    processor = DocumentProcessor()
                    documents = processor.process_directory(settings.uploads_dir)
                    
                    if not documents:
                        st.error("No documents found to process")
                    else:
                        # Save markdown
                        for doc in documents:
                            processor.save_markdown(doc, settings.processed_dir)
                        
                        # Chunk documents
                        chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)
                        chunks = chunker.chunk_documents(documents)
                        st.info(f"✓ Created {len(chunks)} chunks")
                        
                        # Generate embeddings
                        embedder = EmbeddingGenerator(settings.gemini_embedding_model)
                        chunks_with_embeddings = embedder.embed_chunks(chunks)
                        st.info(f"✓ Generated embeddings")
                        
                        # Store in vector database
                        vector_store = VectorStore(settings.collection_name, str(settings.chroma_dir))
                        vector_store.add_chunks(chunks_with_embeddings)
                        
                        st.success(f"✓ Indexed {len(chunks_with_embeddings)} chunks from {len(documents)} documents!")
                        st.balloons()
                        st.cache_resource.clear()
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    if collection_info:
        st.metric("Total Chunks", collection_info['points_count'])
    
    st.divider()
    st.subheader("Query Settings")
    top_k = st.slider("Context Chunks", 1, 10, 5, help="Number of relevant chunks to retrieve")
    max_tokens = st.slider("Max Answer Length", 512, 2048, 1024, step=50)
    
    if error:
        st.divider()
        st.error(f"Error: {error}")
        st.info("💡 This app uses Google Gemini API. Make sure GEMINI_API_KEY is set.")

if error:
    st.error("System not ready. Check sidebar for details.")
else:
    query = st.text_input("Your Question:", placeholder="e.g., What are the main findings of the study?", help="Ask a question about the documents")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_button = st.button("Get Answer", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear", use_container_width=True):
            st.rerun()
    
    if search_button and query:
        try:
            with st.spinner("🔍 Searching documents..."):
                results = retriever.retrieve(query, top_k=top_k)
            
            if not results:
                st.warning("No relevant documents found.")
            else:
                context = retriever.format_context(results)
                
                st.subheader("Answer:")
                with st.spinner("✨"):
                    answer = st.write_stream(generator.generate_answer_stream(query=query, context=context, max_tokens=max_tokens))
                
                st.divider()
                st.subheader("References:")
                
                for i, result in enumerate(results, 1):
                    with st.expander(f"📄 Context {i} - {result['source_file']} (Score: {result['score']:.3f})"):
                        st.text(result['text'])
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    elif search_button:
        st.warning("Please enter a question first!")

