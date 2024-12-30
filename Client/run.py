try:
    # for streamlit >= 1.12.1
    from streamlit.web import bootstrap
except ImportError:
    from streamlit import bootstrap

real_script = "app_client.py"
bootstrap.run(real_script, False, [], {})
