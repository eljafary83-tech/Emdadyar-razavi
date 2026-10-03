from ui.streamlit_app import safe_text_to_html

def test_malicious_output_is_html_escaped():
    assert safe_text_to_html('<img src=x onerror="alert(1)">\ntext') == '&lt;img src=x onerror=&quot;alert(1)&quot;&gt;<br>text'
