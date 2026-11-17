import base64
import hashlib
import requests
import json
import re

import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Diwali Wellness AI", page_icon="🪔")
st.title("🪔 Diwali Wellness AI")
st.caption("Ask about survey habits, adult BMI ranges, or request a funny avatar 🎨")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
TEXT_MODEL = st.secrets.get("TEXT_MODEL", "gpt-4.1-mini")
CLOUDFLARE_MODEL = st.secrets.get("CLOUDFLARE_MODEL", "@cf/black-forest-labs/flux-1-schnell")

@st.cache_data(ttl=60)
def load_data():
    df = pd.read_csv(st.secrets["GOOGLE_SHEET_CSV_URL"])
    df.columns = df.columns.str.strip()
    return df.dropna(how="all").reset_index(drop=True)


def add_bmi(df):
    df = df.copy()
    height = pd.to_numeric(df.get("Height (cm)", pd.Series(index=df.index, dtype=float)), errors="coerce")
    weight = pd.to_numeric(df.get("Weight (kg)", pd.Series(index=df.index, dtype=float)), errors="coerce")
    valid = height.between(100, 250) & weight.between(20, 400)
    df["BMI"] = (weight / (height / 100) ** 2).where(valid).round(1)

    # Exact adult age is preferable. Ambiguous age bands are deliberately excluded.
    if "Age (years)" in df:
        adult = pd.to_numeric(df["Age (years)"], errors="coerce") >= 20
    else:
        age_group = df.get("Age Group", pd.Series("", index=df.index)).fillna("").astype(str)
        def is_adult_band(s):
            s = s.strip().lower()
            if s in ("18-24", "18–24", "under 20", "below 20", "unknown", ""):
                return False
            numbers = re.findall(r"\d+", s)
            return bool(numbers) and int(numbers[0]) >= 20
        adult = age_group.map(is_adult_band)

    def category(bmi, is_adult):
        if pd.isna(bmi):
            return "BMI unavailable"
        if not is_adult:
            return "Adult BMI category unavailable"
        if bmi < 18.5:
            return "Below adult reference range"
        if bmi < 25:
            return "Within adult reference range"
        if bmi < 30:
            return "Above adult reference range"
        return "Obesity BMI range"

    df["BMI Category"] = [category(b, a) for b, a in zip(df["BMI"], adult)]
    return df


try:
    df = add_bmi(load_data())
except Exception as exc:
    st.error(f"Unable to read survey data: {exc}")
    st.stop()

if df.empty:
    st.info("No survey responses yet.")
    st.stop()

# This demo exposes individual rows only to a logged-in administrator.
# Keep ADMIN_PASSWORD configured; a public published CSV is NOT private.
if "ADMIN_PASSWORD" not in st.secrets:
    st.error("Set ADMIN_PASSWORD in Streamlit secrets before using participant data.")
    st.stop()
if not st.session_state.get("authenticated", False):
    password = st.text_input("Admin password", type="password")
    if st.button("Sign in"):
        import hmac
        if hmac.compare_digest(password, str(st.secrets["ADMIN_PASSWORD"])):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()

# The model chooses filters; Python executes them on actual sheet data.
QUERY_TOOL = {
    "type": "function", "name": "query_survey",
    "description": "ALWAYS call first for questions about survey participants or avatar requests. Returns exact matching participant IDs and calculated adult BMI categories. For 'unhealthy' BMI requests use outside, explaining that BMI does not measure overall health.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Participant name or empty string"},
            "gender": {"type": "string", "enum": ["any", "male", "female", "other"]},
            "energy": {"type": "string", "description": "Energy level or empty string"},
            "activity": {"type": "string", "description": "Physical activity or empty string"},
            "bmi_filter": {"type": "string", "enum": ["any", "within", "outside", "below", "above"]}
        },
        "required": ["name", "gender", "energy", "activity", "bmi_filter"],
        "additionalProperties": False
    }, "strict": True
}
AVATAR_TOOL = {
    "type": "function", "name": "generate_avatar",
    "description": "Generate ACTUAL avatar images for participant IDs returned by query_survey. Call ONLY when user requests avatar/image/cartoon. Do not call before query_survey. Use ALL matching IDs unless the user requests fewer.",
    "parameters": {
        "type": "object",
        "properties": {
            "participant_ids": {"type": "array", "items": {"type": "integer"}}
        },
        "required": ["participant_ids"], "additionalProperties": False
    }, "strict": True
}


def query_survey(args):
    selected = df.copy()
    name = args.get("name", "").strip()
    if name and "Participant Name" in selected:
        selected = selected[selected["Participant Name"].fillna("").astype(str).str.contains(re.escape(name), case=False, regex=True)]
    gender = args.get("gender", "any")
    if gender != "any":
        if "Gender" not in selected:
            return selected.iloc[0:0]
        vals = selected["Gender"].fillna("").astype(str).str.strip().str.lower()
        selected = selected[vals.eq(gender) if gender != "other" else ~vals.isin(["male", "female"])]
    for arg, column in (("energy", "Energy Level"), ("activity", "Physical Activity")):
        value = args.get(arg, "").strip()
        if value:
            if column not in selected:
                return selected.iloc[0:0]
            selected = selected[selected[column].fillna("").astype(str).str.casefold() == value.casefold()]
    bmi_filter = args.get("bmi_filter", "any")
    if bmi_filter != "any":
        cat = selected["BMI Category"]
        if bmi_filter == "within":
            selected = selected[cat == "Within adult reference range"]
        elif bmi_filter == "outside":
            selected = selected[cat.isin(["Below adult reference range", "Above adult reference range", "Obesity BMI range"])]
        elif bmi_filter == "below":
            selected = selected[cat == "Below adult reference range"]
        elif bmi_filter == "above":
            selected = selected[cat.isin(["Above adult reference range", "Obesity BMI range"])]
    return selected


def avatar_theme(person, participant_id):
    bmi = person["BMI"]
    gender = str(person.get("Gender", "")).strip().casefold()
    if bmi < 18.5:
        character = "Sparkle Star"
    elif bmi < 25:
        character = "Diwali Dynamo"
    elif bmi < 30:
        character = "Royal Rockstar"
    else:
        character = "Festival Magician"

    costume = ("festive saree or lehenga" if gender == "female" else
               "embroidered kurta or sherwani" if gender == "male" else
               "colorful festive outfit")
    # Stable visual variety: same person gets same design on repeat requests.
    palettes = ["emerald and gold", "violet and silver", "turquoise and coral",
                "ruby and cream", "indigo and saffron", "magenta and teal"]
    roles = ["lantern artist", "rangoli designer", "diya magician",
             "festival musician", "palace storyteller", "light guardian"]
    settings = ["lantern garden", "festive palace courtyard", "Diwali market",
                "rooftop with fairy lights", "colorful rangoli courtyard", "festival stage"]
    seed = hashlib.sha256(str(participant_id).encode()).digest()
    color = palettes[seed[0] % len(palettes)]
    role = roles[seed[1] % len(roles)]
    setting = settings[seed[2] % len(settings)]
    prompt = (
        f"Create ONE distinctive, fictional, friendly 3D cartoon Diwali avatar. "
        f"Character title: {character}. Creative role: {role}. "
        f"Costume: {costume}. Outfit palette: {color}. Scene: {setting}. "
        "Full-body dynamic pose, beautiful stylized character design, expressive smile, "
        "diyas and festive lights, sharp details, square composition. "
        "Make costume, props, pose and setting visually prominent. "
        "Do NOT infer body size, facial appearance or health from BMI. "
        "No real-person likeness, no text, no logos."
    )
    return character, prompt


@st.cache_data(show_spinner=False)
def generate_image(prompt):
    account_id = st.secrets["CLOUDFLARE_ACCOUNT_ID"]
    token = st.secrets["CLOUDFLARE_API_TOKEN"]
    url = (f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
           f"/ai/run/{CLOUDFLARE_MODEL}")
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json={"prompt": prompt},
        timeout=120
    )
    response.raise_for_status()
    # FLUX.1 Schnell REST response is JSON containing a base64 image.
    payload = response.json()
    if not payload.get("success", True):
        raise RuntimeError(f"Cloudflare error: {payload.get('errors')}")
    result = payload.get("result") or {}
    encoded = result.get("image") if isinstance(result, dict) else None
    if not encoded:
        raise RuntimeError("Cloudflare did not return a base64 image. Check model and API response.")
    return base64.b64decode(encoded)


SYSTEM_PROMPT = """You are a friendly Diwali survey assistant. For any survey question, call query_survey first with appropriate filters. Interpret gender words like ladies/women/female or men/male. For avatar or image requests, first call query_survey, then call generate_avatar using the exact participant_ids returned by query_survey. Never merely describe images if images were requested. Never generate images unless explicitly requested. Never invent data. BMI is not a diagnosis and cannot determine who is healthy or healthiest. When 'unhealthy' is clearly about BMI, interpret it as outside the standard adult healthy-weight BMI range and explain this limitation. For under-20 or uncertain age, adult BMI categories are unavailable. Use only the tool results. Keep final replies concise, friendly and playful. Do not shame anyone about their body."""

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("text"):
            st.markdown(msg["text"])
        for img in msg.get("images", []):
            st.image(img["bytes"], caption=img["caption"], width="stretch")

question = st.chat_input("Ask: Generate unique avatars for all female participants")
if question:
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        images = []
        try:
            recent = [{"role": m["role"], "content": m.get("text", "")}
                      for m in st.session_state.messages[-8:]]
            response = client.responses.create(
                model=TEXT_MODEL, instructions=SYSTEM_PROMPT,
                tools=[QUERY_TOOL, AVATAR_TOOL], input=recent)
            # Follow tool calls, returning actual execution results to the LLM.
            selected_ids = set()
            tool_events = []
            for _ in range(5):
                calls = [item for item in response.output if item.type == "function_call"]
                if not calls:
                    break
                outputs = []
                for call in calls:
                    args = json.loads(call.arguments)
                    if call.name == "query_survey":
                        selected = query_survey(args)
                        selected_ids = set(int(i) for i in selected.index)
                        records = []
                        for i, row in selected.iterrows():
                            records.append({
                                "participant_id": int(i),
                                "name": str(row.get("Participant Name", f"Response {i+1}")),
                                "gender": str(row.get("Gender", "")),
                                "bmi": None if pd.isna(row["BMI"]) else float(row["BMI"]),
                                "bmi_category": row["BMI Category"],
                                "energy": str(row.get("Energy Level", "")),
                                "activity": str(row.get("Physical Activity", ""))
                            })
                        result = {"count": len(records), "participants": records}
                    elif call.name == "generate_avatar":
                        requested = args.get("participant_ids", [])
                        valid_ids = list(dict.fromkeys(
                            int(i) for i in requested if int(i) in selected_ids
                        ))
                        limit = int(st.secrets.get("MAX_AVATARS_PER_REQUEST", 5))
                        if len(valid_ids) > limit:
                            result = {"error": f"{len(valid_ids)} participants selected; limit is {limit} images per request. Narrow your request."}
                        else:
                            successes = []
                            failures = []
                            for participant_id in valid_ids:
                                person = df.loc[participant_id]
                                if pd.isna(person["BMI"]):
                                    failures.append({"id": participant_id, "error": "Missing valid height or weight"})
                                    continue
                                character, prompt = avatar_theme(person, participant_id)
                                try:
                                    with st.spinner(f"🎨 Creating avatar {len(successes)+1}/{len(valid_ids)}..."):
                                        img = generate_image(prompt)
                                    caption = f"{person.get('Participant Name', 'Participant')} — {character} 🪔"
                                    images.append({"bytes": img, "caption": caption})
                                    st.image(img, caption=caption, width="stretch")
                                    successes.append({"participant_id": participant_id, "character": character})
                                except Exception as exc:
                                    failures.append({"id": participant_id, "error": str(exc)[:250]})
                            result = {"generated": successes, "failed": failures}
                    else:
                        result = {"error": "Unknown tool"}
                    tool_events.append(result)
                    outputs.append({"type": "function_call_output", "call_id": call.call_id,
                                    "output": json.dumps(result)})
                response = client.responses.create(
                    model=TEXT_MODEL, instructions=SYSTEM_PROMPT,
                    tools=[QUERY_TOOL, AVATAR_TOOL],
                    previous_response_id=response.id, input=outputs)
            answer = response.output_text or (
                f"Generated {len(images)} avatar(s). 🪔" if images else
                "I couldn't complete that request. Please try again."
            )
            st.markdown(answer)
        except Exception as exc:
            answer = f"Sorry, the request failed: {exc}"
            st.error(answer)
    st.session_state.messages.append({"role": "assistant", "text": answer, "images": images})

with st.sidebar:
    st.write(f"📋 Survey responses: {len(df)}")
    st.caption("BMI is calculated in Python when the sheet is read.")
    if st.button("🔄 Refresh data"):
        load_data.clear()
        st.session_state.messages = []
        st.rerun()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()
    if st.button("Log out"):
        st.session_state.authenticated = False
        st.rerun()