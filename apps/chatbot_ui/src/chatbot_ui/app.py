import streamlit as st
import requests
import json
from chatbot_ui.core.config import config


def api_call(method, url, **kwargs):

    def _show_error_popup(message):
        """Show error message as a popup in the top-right corner."""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        response = getattr(requests, method)(url, **kwargs)

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalid response format from server"}

        if response.ok:
            return True, response_data

        return False, response_data

    except requests.exceptions.ConnectionError:
        _show_error_popup("Connection error. Please check your network connection.")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again later.")
        return False, {"message": "Request timeout"}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}


def stream_answer(query):
    response = requests.post(
        f"{config.API_URL}/rag/stream",
        json={"query": query},
        stream=True,
        timeout=(10, 120),
    )
    response.raise_for_status()
    return response


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I assist you today?"}
    ]

if "used_context" not in st.session_state:
    st.session_state.used_context = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Hello! How can I assist you today?")
if prompt:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call backend and stream assistant response token-by-token
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        answer = ""
        try:
            with st.spinner("Thinking..."):
                response = stream_answer(prompt)
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    event = json.loads(line)
                    event_type = event.get("type")
                    if event_type == "token":
                        answer += event.get("content", "")
                        answer_placeholder.markdown(answer)
                    elif event_type == "done":
                        st.session_state.used_context = event.get("used_context", [])

            if not answer:
                answer = "Sorry, I couldn't get a response."
                answer_placeholder.markdown(answer)
        except requests.exceptions.ConnectionError:
            answer = "Connection error. Please check your network connection."
            answer_placeholder.markdown(answer)
        except requests.exceptions.Timeout:
            answer = "The request timed out. Please try again later."
            answer_placeholder.markdown(answer)
        except requests.exceptions.HTTPError as e:
            answer = f"Request failed: {str(e)}"
            answer_placeholder.markdown(answer)
        except Exception as e:
            answer = f"An unexpected error occurred: {str(e)}"
            answer_placeholder.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Sidebar: show suggestions (now reflects latest response)
with st.sidebar:
    st.title("Product suggestions")
    if st.session_state.used_context:
        for item in st.session_state.used_context:
            image_url = item.get("image_url")
            if image_url:
                st.image(image_url, width="stretch")
            description = item.get("description") or ""
            if description:
                first_sentence = description.split(".")[0]
                title = first_sentence.strip()
                if title:
                    trimmed_title = title[:80]
                    if len(title) > 80:
                        trimmed_title += "..."
                    st.markdown(f"**{trimmed_title}**")
            price = item.get("price")
            if price is not None:
                st.markdown(f"**Price:** ${price:,.2f}")
            if description:
                trimmed_desc = description[:200]
                if len(description) > 200:
                    trimmed_desc += "..."
                st.caption(trimmed_desc)
            st.markdown("---")
    else:
        st.caption("Ask about a product to see recommended items here.")
