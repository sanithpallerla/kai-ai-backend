import pytest
from unittest.mock import patch, MagicMock
from features.quizzify.tools import YoutubeTranscriptLoader, Document, VideoTranscriptError, WebPageLoader

@patch('features.quizzify.tools.YoutubeLoader.from_youtube_url')
def test_youtube_transcript_loader(mock_youtube_loader):
    mock_url = "https://www.youtube.com/watch?v=example"
    mock_video_urls = [mock_url]
    
    # Mock document
    mock_document = MagicMock(spec=Document)
    mock_document.metadata = {'author': 'Author', 'title': 'Title', 'length': 100}
    
    # Mock the loader's load method
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = [mock_document]
    mock_youtube_loader.return_value = mock_loader_instance
    
    loader  = YoutubeTranscriptLoader(video_urls=mock_video_urls)
    documents = loader.load()
    
    assert len(documents) == 1
    assert documents[0].metadata['author'] == 'Author'


@patch('features.quizzify.tools.WebBaseLoader')
def test_webpage_loader(mock_web_loader):
    mock_url = "https://www.apple.com/apple-intelligence/?mtid=209251kg40341&aosid=p238&mnid=s2v2gHuUw-dc_mtid_209251kg40341_pcrid_702087218873_pgrid_160690191302_pexid__&cid=wwa-ca-kwgo-features-slid-----"
    mock_web_urls = [mock_url]
    
    # Mock document
    mock_document = MagicMock(spec=Document)
    
    # Mock the loader's load method
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = [mock_document]
    mock_web_loader.return_value = mock_loader_instance
    
    loader = WebPageLoader(web_urls=mock_web_urls)
    documents = loader.load()
    
    assert len(documents) == 1