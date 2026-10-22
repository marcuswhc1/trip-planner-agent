import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, List, Union

import gradio as gr
import requests
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


###############
# Configuration
###############
ENV_FILE = Path(__file__).with_name(".env")
with ENV_FILE.open(encoding="utf-8-sig") as env_file:
    for line_number, line in enumerate(env_file, start=1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        name, separator, value = entry.partition("=")
        if not separator or not name.strip():
            raise ValueError(f"Invalid .env entry on line {line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(name.strip(), value)

TAVILY_API_KEY  = os.environ.get("TAVILY_API_KEY")
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
HOLIDAYS_API    = os.environ.get("HOLIDAYS_API")
OUTPUT_DIR      = Path("agent_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


##########################################################################
# Tools  (plain @tool functions — LangChain / LangGraph understands these)
##########################################################################
@tool
def currency_converter(conversions: List[dict]) -> dict:
    """
    Convert multiple currency amounts concurrently using the Frankfurter API.

    Each item in `conversions` must have:
      - 'from'   : source currency code, e.g. 'USD'
      - 'to'     : target currency code, e.g. 'SGD'
      - 'amount' : numeric amount to convert

    Returns a dict mapping 'FROM->TO' to the converted amount (or an error string).
    """
    def fetch_one(c):
        base, target, amount = c["from"], c["to"], c["amount"]
        url = f"https://api.frankfurter.app/latest?amount={amount}&from={base}&to={target}"
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            rate = r.json()["rates"].get(target)
            return f"{base}->{target}", rate if rate is not None else "no rate returned"
        except Exception as e:
            return f"{base}->{target}", f"Error: {e}"

    with ThreadPoolExecutor() as ex:
        return dict(ex.map(fetch_one, conversions))


@tool
def fetch_public_holidays(country: str, year: int) -> Union[List[str], str]:
    """
    Fetch public holidays for a country.

    Args:
        country: 2-letter ISO country code, e.g. 'SG', 'JP', 'US'
        year:    4-digit year, e.g. 2026
    """
    url = f"{HOLIDAYS_API}/PublicHolidays/{year}/{country}"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        holidays = r.json()
        if not holidays:
            return f"No holidays found for {country} in {year}."
        return [f"- {h['date']}: {h['localName']}" for h in holidays]
    except requests.exceptions.HTTPError:
        return f"Invalid country code '{country}' or year '{year}'."
    except requests.RequestException as e:
        return f"Error fetching holidays: {e}"


@tool
def get_current_weather(city: str) -> str:
    """
    Get the current weather for a city.

    Args:
        city: City name, e.g. 'Singapore', 'Tokyo'
    """
    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={WEATHER_API_KEY}&units=metric"
    )
    r = requests.get(url)
    if r.status_code == 200:
        d = r.json()
        desc  = d["weather"][0]["description"]
        temp  = d["main"]["temp"]
        hum   = d["main"]["humidity"]
        wind  = d["wind"]["speed"]
        return (
            f"The weather in {city} is {desc} with a temperature of "
            f"{temp}°C, humidity of {hum}%, and wind speed of {wind} m/s."
        )
    return f"Error {r.status_code}: {r.json()}"


@tool
def web_search(query: str) -> str:
    """
    Search the web for tourist attractions, landmarks, and things to do in a
    specific region or city.

    Args:
        query: City or region name plus keywords, e.g. 'Tokyo attractions'
    """
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": f"top tourist attractions and landmarks in {query}",
        "search_depth": "basic",
        "include_answer": True,
        "max_results": 5,
    }
    try:
        resp = requests.post("https://api.tavily.com/search", json=payload, timeout=20).json()
        if resp.get("answer"):
            return resp["answer"]
        results = resp.get("results", [])
        if not results:
            return f"No attractions found for: {query}"
        lines = [f"Search results for attractions in {query}:\n"]
        for i, res in enumerate(results, 1):
            lines.append(f"{i}. {res['title']}\nURL: {res['url']}\nSnippet: {res['content']}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"An error occurred while searching: {e}"


@tool
def search_flights(
    origin: str,
    destination: str,
    date: str,
    passengers: int = 1,
    currency: str = "SGD",
) -> str:
    """
    Search Google Flights (via SerpApi) for one-way flights on a given date.
    Returns up to 5 options with date, route, airline, flight number,
    departure/arrival times, stops, duration and price.
    For a round trip, call twice — once per direction.

    Args:
        origin:      Departure airport IATA code, e.g. 'SIN'
        destination: Arrival airport IATA code,   e.g. 'HND'
        date:        Travel date in YYYY-MM-DD format
        passengers:  Number of adult passengers (default 1)
        currency:    Currency code for prices   (default 'SGD')
    """
    if not SERPAPI_API_KEY:
        return "SERPAPI_API_KEY is not set — flight search unavailable."

    passengers = passengers or 1
    currency   = currency   or "SGD"
    params = {
        "engine":         "google_flights",
        "departure_id":   origin,
        "arrival_id":     destination,
        "outbound_date":  date,
        "type":           2,          # one-way
        "adults":         passengers,
        "currency":       currency,
        "hl":             "en",
        "api_key":        SERPAPI_API_KEY,
    }
    try:
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=60)
        data = resp.json()
    except Exception as e:
        return f"Flight search failed: {e}"

    if data.get("error"):
        return f"Flight search error: {data['error']}"

    options = ((data.get("best_flights") or []) + (data.get("other_flights") or []))[:5]
    if not options:
        return f"No flights found for {origin}->{destination} on {date}."

    rows = []
    for opt in options:
        legs = opt.get("flights") or []
        if not legs:
            continue
        first, last = legs[0], legs[-1]
        mins = opt.get("total_duration") or 0
        rows.append({
            "Date":     date,
            "Route":    f"{first['departure_airport'].get('id','')}->{last['arrival_airport'].get('id','')}",
            "Airline":  ", ".join(dict.fromkeys(l.get("airline","") for l in legs)),
            "Flight":   ", ".join(l.get("flight_number","") for l in legs),
            "Departs":  first["departure_airport"].get("time",""),
            "Arrives":  last["arrival_airport"].get("time",""),
            "Stops":    len(legs) - 1,
            "Duration": f"{mins//60}h {mins%60}m" if mins else "",
            "Price":    f"{currency} {opt['price']}" if opt.get("price") else "N/A",
        })

    link = (data.get("search_metadata") or {}).get("google_flights_url","")
    return json.dumps({"flights": rows, "google_flights_url": link}, ensure_ascii=False)


#######################################################
# Word export tool (landscape A4, minimal margins/padding)
#######################################################
_INLINE_MARKUP = re.compile(r"(\*\*.+?\*\*|\*.+?\*)")

# Fixed itinerary layout: (header, dict key, column width in inches)
ITINERARY_COLUMNS = [
    ("Date",          "date",          0.9),
    ("Itinerary",     "itinerary",     5.0),
    ("Location",      "location",      1.7),
    ("Transport",     "transport",     2.0),
    ("Accommodation", "accommodation", 1.59),
]

# Fixed summary layout shown above the itinerary
SUMMARY_COLUMNS = [
    ("Item",    "item",    2.2),
    ("Details", "details", 7.0),
    ("Amount",  "amount",  1.99),
]


def _text(value) -> str:
    """Lists become one line per item; everything else is stringified."""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v) for v in value)
    return "" if value is None else str(value)


def _add_runs(paragraph, text: str, bold: bool = False) -> None:
    """Add text to a paragraph, honouring **bold** and *italic* markup."""
    for part in _INLINE_MARKUP.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            run = paragraph.add_run(part[1:-1])
            run.italic = True
            run.bold = bold or None
        else:
            paragraph.add_run(part).bold = bold or None


def _fill_cell(cell, text: str, bold: bool = False) -> None:
    """Write multi-line text into a cell; lines starting with '- ' become bullets."""
    lines = _text(text).split("\n")
    for i, line in enumerate(lines):
        p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        if line.startswith("- "):
            p.style = "List Bullet"
            line = line[2:]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        _add_runs(p, line, bold=bold)


def _shade_cell(cell, fill: str) -> None:
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def _set_cell_padding(table, top=20, bottom=20, left=60, right=60) -> None:
    """Cell padding in twips (1/20 pt)."""
    tbl_pr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for side, val in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    look = tbl_pr.find(qn("w:tblLook"))
    if look is not None:
        look.addprevious(mar)
    else:
        tbl_pr.append(mar)


def _add_table(doc, headers: list, rows: list, widths: list = None) -> None:
    n_cols = len(headers) or max(len(r) for r in rows)
    table = doc.add_table(rows=0, cols=n_cols)
    table.style = "Table Grid"
    _set_cell_padding(table)
    if widths:
        table.autofit = False

    def add_row(values, header=False):
        cells = table.add_row().cells
        for i, c in enumerate(cells):
            if widths:
                c.width = Inches(widths[i])
            _fill_cell(c, values[i] if i < len(values) else "", bold=header)
            if header:
                _shade_cell(c, "D9D9D9")

    if headers:
        add_row(headers, header=True)
    for row in rows:
        add_row(list(row))


def _add_heading(doc, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    _add_runs(p, text, bold=True)


@tool
def export_to_word(
    title: str,
    itinerary: List[dict],
    summary: List[dict] = None,
    sections: List[dict] = None,
) -> str:
    """
    Export the trip plan to a Word (.docx) file that can be uploaded to Google Docs
    (A4 landscape, minimal margins, tight table padding).
    The day-by-day plan is ALWAYS rendered as ONE table with exactly 5 columns:
    Date | Itinerary | Location | Transport | Accommodation, one row per day.
    Never split morning/afternoon/evening into separate columns: put every time slot
    of the day inside that day's single 'itinerary' cell, one line each.

    Args:
        title: Document title, e.g. 'Guangzhou 4-11 Nov 2026'
        itinerary: One dict per day with keys:
          - 'date'          : e.g. '**4 Nov, Wed**'
          - 'itinerary'     : all activities that day, one per line separated by '\\n'
          - 'location'      : city/area for that day
          - 'transport'     : how the user gets around that day
          - 'accommodation' : where the user stays that night
        summary: Optional table shown ABOVE the itinerary, with exactly 3 columns
          Item | Details | Amount. Use it for flight cost, accommodation cost,
          estimated activity & transport budget (itemised with '- ' bullets in
          details), total estimated trip cost, work leave required and any other notes and tips. One dict per row:
          - 'item'    : e.g. '**Flight Cost (2 adults)**'
          - 'details' : explanation / breakdown, '\\n' separated
          - 'amount'  : e.g. '**SGD 2,049**' (leave empty if not a cost)
        sections: Optional extra blocks shown after the itinerary (e.g. paid event
          tickets). Each is a dict with optional keys:
          - 'heading'    : bold heading
          - 'paragraphs' : list of text lines
          - 'headers'    : list of column names for an extra table
          - 'rows'       : list of rows (lists of cell strings)

        In any cell or paragraph: '\\n' = new line, '- ' = bullet, '**x**' = bold, '*x*' = italic.

    Returns the path of the saved file.
    """
    doc = Document()
    page = doc.sections[0]
    page.orientation = WD_ORIENT.LANDSCAPE
    page.page_width, page.page_height = Inches(11.69), Inches(8.27)
    page.top_margin = page.bottom_margin = Inches(0.25)
    page.left_margin = page.right_margin = Inches(0.25)
    page.header_distance = page.footer_distance = Inches(0)

    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10)
    normal.paragraph_format.space_before = Pt(0)

    _add_heading(doc, f"**{title}**")

    if summary:
        _add_table(
            doc,
            [h for h, _, _ in SUMMARY_COLUMNS],
            [[row.get(key, "") for _, key, _ in SUMMARY_COLUMNS] for row in summary],
            [w for _, _, w in SUMMARY_COLUMNS],
        )
        doc.add_paragraph()  # keeps the two tables from merging

    for sec in sections or []:
        if sec.get("heading"):
            _add_heading(doc, sec["heading"])
        for text in sec.get("paragraphs") or []:
            _add_runs(doc.add_paragraph(), _text(text))
        headers, rows = sec.get("headers") or [], sec.get("rows") or []
        if headers or rows:
            _add_table(doc, headers, rows)

    if itinerary:
        _add_table(
            doc,
            [h for h, _, _ in ITINERARY_COLUMNS],
            [[day.get(key, "") for _, key, _ in ITINERARY_COLUMNS] for day in itinerary],
            [w for _, _, w in ITINERARY_COLUMNS],
        )

    safe = re.sub(r"[^\w\-]+", "_", title).strip("_")[:50] or "trip_plan"
    path = OUTPUT_DIR / f"{safe}_{int(time.time())}.docx"
    doc.save(path)
    return f"Word document saved to {path}"

########################################
# Store all tools into a tools registry
########################################
ALL_TOOLS = [
    currency_converter,
    fetch_public_holidays,
    get_current_weather,
    web_search,
    search_flights,
    export_to_word
]


#######################################
# Model configuration and tool binding
#######################################
llm = ChatOllama(
    model=os.environ.get("LLM_MODEL"),
    base_url=os.environ.get("LLM_MODEL_BASE_URL"),
    num_ctx=8192,
).bind_tools(ALL_TOOLS)

#################################
# Agent state and router function
#################################
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def call_llm(state: AgentState) -> dict:
    """Node: send messages to the LLM and return its reply."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Edge: go to tools if the LLM made a tool call, else end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

#################
# Build the graph
#################
TOOL_NODE = ToolNode(ALL_TOOLS)

graph_builder = StateGraph(AgentState)
graph_builder.add_node("llm",   call_llm)
graph_builder.add_node("tools", TOOL_NODE)

graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "llm")   # after tools, always go back to LLM

agent_graph = graph_builder.compile()


############
# Gradio UI
############
SYSTEM_PROMPT = (
    "You are a helpful travel planning assistant. "
    "You have access to tools for searching flights, checking weather, "
    "fetching public holidays, converting currencies, searching attractions, "
    "booking air tickets, and exporting results to Word documents. "
    "Always use the most relevant tool(s) to answer the user's question. "
    "When the user asks for a Word document or an itinerary file, call export_to_word. "
    "Its itinerary must have one row per day with exactly these columns: "
    "Date, Itinerary, Location, Transport, Accommodation. "
    "Never create separate Morning/Afternoon/Evening columns; put all of a day's "
    "activities in that day's single Itinerary cell, one line per activity, "
    "each starting with a time (e.g. '0930: ...'). "
    "Put flight cost, accommodation cost, estimated activity & transport budget, "
    "total estimated trip cost and work leave required and additional tips in export_to_word's `summary` "
    "table (columns Item, Details, Amount), not as free text below the itinerary."
)
 
 
def to_lc_messages(history: list) -> list:
    """Gradio history -> LLM messages, skipping the tool-call display bubbles."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        if m.get("metadata"):          # tool bubble, not real conversation
            continue
        content = m["content"]
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )

        msgs.append({"role": m["role"], "content": content})
    return msgs


with gr.Blocks(title="Trip Planning Agent") as demo:
    gr.Markdown("## ✈️ Trip Planning Agent")

    chatbot  = gr.Chatbot(
        value=[],
        label="Trip Agent",
        height=500,
        placeholder="<h2><center>Welcome! How can I help you plan your trip today?</center></h2>",
    )
    msg_box  = gr.Textbox(placeholder="Ask me anything about your trip…", show_label=False)
    download = gr.File(label="Download Word document", visible=False)
    clear    = gr.Button("New conversation")

    # def respond(message, history):
    #     new_history, doc_path = chat(message, history)
    #     dl_update = gr.update(value=doc_path, visible=bool(doc_path))
    #     return new_history, "", dl_update

    def respond(message, history):
        files_before = set(OUTPUT_DIR.glob("*.docx"))
        lc_messages = to_lc_messages(history) + [{"role": "user", "content": message}]
    
        history = history + [{"role": "user", "content": message}]
        yield history, "", gr.update()
    
        for update in agent_graph.stream({"messages": lc_messages}, stream_mode="updates"):
            for node_output in update.values():
                for msg in node_output["messages"]:
    
                    # 1) The LLM decided to call tool(s) -> show a pending bubble
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            args = json.dumps(tc["args"], indent=2, ensure_ascii=False)
                            history.append({
                                "role": "assistant",
                                "content": f"```json\n{args}\n```",
                                "metadata": {
                                    "title": f"🛠️ Calling {tc['name']}",
                                    "id": tc["id"],
                                    "status": "pending",
                                },
                            })
    
                    # 2) A tool finished -> attach its result to the matching bubble
                    elif isinstance(msg, ToolMessage):
                        result = str(msg.content)
                        if len(result) > 1500:
                            result = result[:1500] + "\n… (truncated)"
                        for m in history:
                            meta = m.get("metadata") or {}
                            if meta.get("id") == msg.tool_call_id:
                                m["content"] += f"\n\n**Result:**\n```\n{result}\n```"
                                meta["status"] = "done"
                                break
    
                    # 3) Final answer from the LLM
                    elif isinstance(msg, AIMessage):
                        history.append({"role": "assistant", "content": msg.content or ""})
    
            yield history, "", gr.update()

            # Only offer Word files created during THIS turn
            new_files = sorted(set(OUTPUT_DIR.glob("*.docx")) - files_before, key=os.path.getmtime)
            if new_files:
                yield history, "", gr.update(value=str(new_files[-1]), visible=True)

    def reset():
        return [], gr.update(value=None, visible=False)

    msg_box.submit(respond, [msg_box, chatbot], [chatbot, msg_box, download])
    clear.click(reset, None, [chatbot, download])

demo.queue(default_concurrency_limit=4).launch()