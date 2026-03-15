import streamlit as st
import requests
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

    # Call backend and show assistant response with spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            success, response_data = api_call(
                "post",
                f"{config.API_URL}/rag",
                json={"query": prompt},
            )
            if success and "answer" in response_data:
                answer = response_data["answer"]
                st.session_state.used_context = response_data.get("used_context", [])
            else:
                detail = response_data.get(
                    "detail",
                    response_data.get(
                        "message",
                        "Sorry, I couldn't get a response.",
                    ),
                )
                answer = (
                    detail
                    if isinstance(detail, str)
                    else "; ".join(d.get("msg", str(d)) for d in detail)
                    if isinstance(detail, list)
                    else str(detail)
                )
            st.markdown(answer)
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
