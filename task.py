import json
import urllib.request
import subprocess
import requests
from fastapi import FastAPI, HTTPException, Query
import os
import sqlite3
import markdown
from bs4 import BeautifulSoup
from PIL import Image
import speech_recognition as sr
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from datetime import datetime
import numpy as np
import glob
from pydantic import BaseModel

app = FastAPI()


AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise RuntimeError("AIPROXY_TOKEN is missing. Set it in environment variables.")


client = OpenAI(api_key=AIPROXY_TOKEN)

# ✅ CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Tools Definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "script_runner",
            "description": "Install a package and run a script from a URL with provided arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script_url": {
                        "type": "string",
                        "description": "The URL of the script to run."
                    },
                    "args": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Arguments to pass to the script."
                    }
                },
                "required": ["script_url", "args"]
            }
        }
    }
]

@app.get("/")
def home():
    return {"message": "Yay! TDS Tuesday is awesome."}

@app.get("/read")
def read_file(path: str):
    """Reads a file from the given path."""
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File doesn't exist")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

class TaskRequest(BaseModel):
    task: str

@app.post("/run")
def run_task(task: str):
    """Handles various tasks using LLM."""
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": task}],
        "tools": tools,
        "tool_choice": "auto"
    }

    response = requests.post(url, headers=headers, json=data)

    try:
        response_data = response.json()

        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            if "tool_calls" in choice and choice["tool_calls"]:
                tool_call = choice["tool_calls"][0]
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])

                # ✅ If the function is "script_runner", download and run the script
                if function_name == "script_runner":
                    return download_and_run_script(arguments["script_url"], arguments.get("args", []))

        return response_data  # Return raw response if no tool calls
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing API response: {str(e)}")

def download_and_run_script(script_url, args):
    """Downloads a script from a URL and executes it."""
    try:
        script_path = os.path.join(os.getcwd(), "datagen.py")

        # ✅ Download the script
        response = requests.get(script_url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to download script")

        # ✅ Save it locally
        with open(script_path, "w", encoding="utf-8") as file:
            file.write(response.content.decode("utf-8"))  # Ensure proper decoding

        # ✅ Run the script
        command = ["python", script_path] + args
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        return {"message": "Script executed successfully", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Script execution failed: {e.stderr}")

@app.post("/run_datagen")
def run_datagen(email: str):
    """Runs datagen.py and ensures the 'data' folder is saved in the current directory."""
    working_directory = os.getcwd()  # Get current working directory
    script_path = os.path.join(working_directory, "datagen.py")
    data_destination = os.path.join(working_directory, "data")  # Correct output folder

    try:
        # Step 1: Run datagen.py with --root argument
        command = ["python", script_path, "--root", data_destination, email]
        result = subprocess.run(command, capture_output=True, text=True, cwd=working_directory)

        return {
            "message": "Script executed successfully",
            "output": result.stdout,
            "error": result.stderr,
            "saved_in": data_destination if os.path.exists(data_destination) else "Not found"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running script: {str(e)}")

@app.post("/task-a2")
def run_task_a2(task: str):
    """Handles various tasks dynamically using LLM."""
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": task}],
        "tools": tools,
        "tool_choice": "auto"
    }

    response = requests.post(url, headers=headers, json=data)

    try:
        response_data = response.json()

        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            if "tool_calls" in choice and choice["tool_calls"]:
                tool_call = choice["tool_calls"][0]
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])

                if function_name == "format_markdown":
                    return format_markdown(arguments["input_file"])

        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing API response: {str(e)}")

def format_markdown(input_file: str):
    """Formats a markdown file using Prettier and saves it."""
    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail=f"File not found: {input_file}")

    try:
        command = ["npx", "prettier@3.4.2", "--write", input_file]
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        return {"message": "Markdown file formatted successfully", "input_file": input_file, "output": result.stdout}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Formatting failed: {e.stderr}")



@app.post("/task-a3 ")
def run_task(task: str):
    """Handles various tasks dynamically using LLM."""
    
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": task}],
        "tools": tools, 
        "tool_choice": "auto"
    }

    response = requests.post(url, headers=headers, json=data)

    try:
        response_data = response.json()

        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            if "tool_calls" in choice and choice["tool_calls"]:
                tool_call = choice["tool_calls"][0]
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])

                # ✅ Handle Task A3 dynamically
                if function_name == "count_weekday_occurrences":
                    return count_weekday_occurrences(arguments["input_file"], arguments["day"], arguments["output_file"])

        return response_data  # Return raw response if no tool calls
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing API response: {str(e)}")


def count_weekday_occurrences(input_file: str, day: str, output_file: str):
    """Counts occurrences of a specific weekday in a file and writes the count to output_file."""
    
    weekday_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
        "सोमवार": 0, "मंगलवार": 1, "बुधवार": 2, "गुरुवार": 3, "शुक्रवार": 4, "शनिवार": 5, "रविवार": 6,  # Hindi
        "திங்கள்": 0, "செவ்வாய்": 1, "புதன்": 2, "வியாழன்": 3, "வெள்ளி": 4, "சனி": 5, "ஞாயிறு": 6  # Tamil
    }

    if day not in weekday_map:
        raise HTTPException(status_code=400, detail=f"Invalid day: {day}")

    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail=f"File not found: {input_file}")

    def parse_date(date_string):
        formats = ["%Y-%m-%d", "%Y/%m/%d %H:%M:%S"]
        for fmt in formats:
            try:
                return datetime.strptime(date_string.strip(), fmt)
            except ValueError:
                continue
        return None

    count = 0
    with open(input_file, "r") as file:
        for date in file:
            parsed_date = parse_date(date)
            if parsed_date and parsed_date.weekday() == weekday_map[day]:
                count += 1

    with open(output_file, "w") as output_file_obj:
        output_file_obj.write(str(count))

    return {
        "task": f"Counted {day} occurrences",
        "weekday": day,
        "count": count,
        "output_file": output_file
    }




from pathlib import Path



@app.post("/task-a4")
def task_a4(task: str = Query(..., description="Task description")):
    """
    Task A4: Parses a natural language task, extracts file names, sorts contacts by last_name,
    and saves the sorted data to an output file.
    """

    try:
        # ✅ Strong LLM Prompt to extract file paths
        prompt = f"""
        You are an intelligent assistant that extracts file names from a task description. 
        Your job is to analyze the text and identify:
        - The input file: The JSON file containing the unsorted contacts.
        - The output file: The JSON file where sorted contacts should be saved.
        
        The task may be written in different ways or even in different languages. Extract the filenames
        correctly and return them in JSON format.

        Example input:
        "Sort contacts from /data/contacts.json and save to /data/sorted.json"

        Expected output:
        {{
          "input_file": "/data/contacts.json",
          "output_file": "/data/sorted.json"
        }}

        Now extract filenames from this task:
        Task: "{task}"
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        extracted_data = json.loads(response.choices[0].message.content)

        input_file = extracted_data.get("input_file")
        output_file = extracted_data.get("output_file")

        # ✅ Validate extracted file paths
        if not input_file or not output_file:
            raise ValueError("Failed to extract input_file or output_file from the task.")

        # ✅ Check if the input file exists
        if not Path(input_file).is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {input_file}")

        # ✅ Read the contacts file
        with open(input_file, "r") as file:
            content = file.read().strip()
            if not content:
                raise ValueError("Contacts file is empty")

            contacts = json.loads(content)

        # ✅ Ensure contacts is a list
        if not isinstance(contacts, list):
            raise ValueError("Invalid contacts format. Expected a list of dictionaries.")

        # ✅ Sort contacts by last_name, then first_name
        contacts_sorted = sorted(
            contacts,
            key=lambda x: (
                x.get("last_name", "").strip().lower() if x.get("last_name") else "",
                x.get("first_name", "").strip().lower() if x.get("first_name") else ""
            )
        )

        # ✅ Save sorted contacts to the output file
        with open(output_file, "w") as file:
            json.dump(contacts_sorted, file, indent=2)

        return {
            "message": "Contacts sorted successfully",
            "input_file": input_file,
            "output_file": output_file
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in contacts file")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/task-a5")
def task_a5(task: str = Query(..., description="Task description")):
    """
    Task A5: Extracts the first line of the 10 most recent log files in a directory
    and saves them to a specified output file.
    """

    try:
        # ✅ Strong LLM Prompt to extract directory and output file
        prompt = f"""
        You are an intelligent assistant that extracts directory and file paths from a task description.
        Your job is to analyze the text and identify:
        - The directory containing log files.
        - The output file where the extracted first lines should be saved.

        The task may be written in different ways or even in different languages. Extract the paths
        correctly and return them in JSON format.

        Example input:
        "Extract first lines from the latest 10 logs in /data/logs/ and save them to /data/logs-recent.txt"

        Expected output:
        {{
          "logs_directory": "/data/logs/",
          "output_file": "/data/logs-recent.txt"
        }}

        Now extract paths from this task:
        Task: "{task}"
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        extracted_data = json.loads(response.choices[0].message.content)

        logs_dir = extracted_data.get("logs_directory")
        output_file = extracted_data.get("output_file")

        # ✅ Validate extracted paths
        if not logs_dir or not output_file:
            raise ValueError("Failed to extract logs_directory or output_file from the task.")

        # ✅ Ensure the directory exists
        if not Path(logs_dir).is_dir():
            raise HTTPException(status_code=404, detail=f"Logs directory not found: {logs_dir}")

        # ✅ Get the 10 most recent .log files
        log_files = sorted(
            glob.glob(os.path.join(logs_dir, "*.log")),
            key=os.path.getmtime,
            reverse=True
        )[:10]  # Pick the 10 most recent

        log_lines = []

        for log_file in log_files:
            try:
                with open(log_file, "r") as file:
                    first_line = file.readline().strip()
                    if first_line:
                        log_lines.append(first_line)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading {log_file}: {str(e)}")

        # ✅ Write to output file
        with open(output_file, "w") as out_file:
            out_file.write("\n".join(log_lines) + "\n")

        return JSONResponse(content={"message": "Logs processed successfully", "output_file": output_file})

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON response from LLM")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/task-a6")
def task_a6(task: str = Query(..., description="Task description")):
    """
    Task A6: Finds Markdown (.md) files, extracts the first H1 header, and creates an index file.
    """

    try:
        # ✅ Strong LLM Prompt to extract directory and output file
        prompt = f"""
        You are an intelligent assistant that extracts directory and file paths from a task description.
        Your job is to analyze the text and identify:
        - The directory containing Markdown (.md) files.
        - The output file where the extracted headers should be saved.

        The task may be written in different ways or even in different languages. Extract the paths
        correctly and return them in JSON format.

        Example input:
        "Extract the first # header from each Markdown file in /data/docs/ and save to /data/docs/index.json"

        Expected output:
        {{
          "docs_directory": "/data/docs/",
          "output_file": "/data/docs/index.json"
        }}

        Now extract paths from this task:
        Task: "{task}"
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        extracted_data = json.loads(response.choices[0].message.content)

        docs_dir = extracted_data.get("docs_directory")
        output_file = extracted_data.get("output_file")

        # ✅ Validate extracted paths
        if not docs_dir or not output_file:
            raise ValueError("Failed to extract docs_directory or output_file from the task.")

        # ✅ Ensure the directory exists
        if not Path(docs_dir).is_dir():
            raise HTTPException(status_code=404, detail=f"Docs directory not found: {docs_dir}")

        md_files = glob.glob(os.path.join(docs_dir, "*.md"))
        index_data = {}

        for md_file in md_files:
            try:
                with open(md_file, "r", encoding="utf-8") as file:
                    for line in file:
                        line = line.strip()
                        if line.startswith("# "):  # First occurrence of H1
                            filename = os.path.basename(md_file)  # Extract filename
                            index_data[filename] = line[2:]  # Remove '# ' prefix
                            break  # Stop after the first H1
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading {md_file}: {str(e)}")

        # ✅ Write index data to JSON file
        with open(output_file, "w", encoding="utf-8") as out_file:
            json.dump(index_data, out_file, indent=2)

        return JSONResponse(content={"message": "Markdown index created successfully", "output_file": output_file})

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON response from LLM")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class EmailExtractionRequest(BaseModel):
    task_description: str

@app.post("/task-a7")
def task_a7(request: EmailExtractionRequest):
    """
    You are a language understanding LLM capable of accurately extracting specific pieces of information from unstructured text.
    Your mission is to extract the sender's email address from an email file containing the full content of an email.
    Follow these precise steps:
    1. Locate the plain text email file at '/data/email.txt'.
    2. Read the entire content of the email.
    3. Pass the content to an LLM (GPT-4o-mini) with a clear and structured prompt to extract the sender’s email address.
    4. Ensure the output is a valid email address in the standard format (e.g., example@example.com).
    5. Write the extracted email address to the file '/data/email-sender.txt'.
    6. Return the extracted email address in the response JSON.
    Ensure robust error handling for file access, text parsing, and LLM failures.
    """
    email_file = os.path.join(DATA_DIR, "email.txt")
    output_file = os.path.join(DATA_DIR, "email-sender.txt")

    if not os.path.exists(email_file):
        raise HTTPException(status_code=404, detail="Email file not found")

    try:
        # Step 1: Read email content
        with open(email_file, "r", encoding="utf-8") as file:
            email_content = file.read()

        # Step 2: LLM prompt to extract sender's email
        prompt = f"""
        Extract the sender's email address from the following raw email content:

        {email_content}

        - Provide only the sender’s email address in standard format (e.g., user@example.com).
        - Do not include any other text, explanations, or symbols. Return only the email address.
        - If multiple email addresses are present, return the primary sender's address.
        """

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        extracted_email = response.choices[0].message.content.strip()

        # Step 3: Save extracted email address to file
        with open(output_file, "w", encoding="utf-8") as out_file:
            out_file.write(extracted_email)

        return {
            "message": "Sender's email address extracted successfully",
            "email": extracted_email,
            "output_file": output_file
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class CardExtractionRequest(BaseModel):
    task_description: str

@app.post("/task-a8")
def task_a8(request: CardExtractionRequest):
    """
    You are an expert image-processing and text-extraction LLM capable of analyzing images with OCR (Optical Character Recognition) and refining outputs using advanced language understanding.
    Your mission is to accurately extract a credit card number from an image file and format it properly.
    Follow these precise steps:
    1. Locate the image file at '/data/credit-card.png'. This image contains a visible credit card number.
    2. Use the pytesseract OCR library to extract text from the image.
    3. Construct a detailed prompt for an LLM (GPT-4o-mini) to identify, clean, and return only the card number from the extracted text.
    4. Ensure the output consists strictly of digits, without spaces, dashes, or other characters.
    5. Write the final card number to the file '/data/credit-card.txt'.
    6. Return the card number as JSON for verification purposes.
    Ensure robust error handling for file access, OCR issues, and LLM API failures.
    """
    image_file = os.path.join(DATA_DIR, "credit-card.png")
    output_file = os.path.join(DATA_DIR, "credit-card.txt")

    if not os.path.exists(image_file):
        raise HTTPException(status_code=404, detail="Credit card image not found")

    try:
        # Step 1: Load image and extract text using OCR
        image = Image.open(image_file)
        extracted_text = pytesseract.image_to_string(image)

        # Step 2: LLM prompt to clean and extract card number
        prompt = f"""
        Extract the credit card number from the following text snippet obtained through OCR:

        {extracted_text}

        - Provide only the card number as a sequence of digits with no spaces, dashes, or extra characters.
        - If multiple possible card numbers appear, return the most complete and likely correct one.
        - Do not include any other text, explanations, or symbols. Only output the digits of the card number.
        """

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        card_number = response.choices[0].message.content.strip()

        # Step 3: Save extracted card number to file
        with open(output_file, "w", encoding="utf-8") as out_file:
            out_file.write(card_number)

        return {
            "message": "Credit card number extracted successfully",
            "card_number": card_number,
            "output_file": output_file
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# File paths
COMMENTS_FILE = "/data/comments.txt"
OUTPUT_FILE = "/data/comments-similar.txt"


class TaskRequest(BaseModel):
    task_description: str

@app.post("/task-a9")
def find_similar_comments(request: TaskRequest):
    """Finds the two most similar comments from /data/comments.txt using embeddings."""
    
    input_file = os.path.join(DATA_DIR, "comments.txt")
    output_file = os.path.join(DATA_DIR, "comments-similar.txt")

    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail="Comments file not found")

    try:
        with open(input_file, "r", encoding="utf-8") as file:
            comments = [line.strip() for line in file.readlines() if line.strip()]

        if len(comments) < 2:
            raise HTTPException(status_code=400, detail="Not enough comments for similarity comparison")

        # Generate embeddings
        response = openai.Embedding.create(input=comments, model="text-embedding-ada-002")
        embeddings = [item["embedding"] for item in response["data"]]

        # Compute cosine similarity
        similarities = np.inner(embeddings, embeddings)
        np.fill_diagonal(similarities, -1)  # Ignore self-similarity

        # Find most similar pair
        i, j = np.unravel_index(np.argmax(similarities), similarities.shape)
        similar_pair = [comments[i], comments[j]]

        # Save to file
        with open(output_file, "w", encoding="utf-8") as out_file:
            out_file.write(f"{similar_pair[0]}\n{similar_pair[1]}")

        return {"message": "Most similar comments identified successfully", "comments": similar_pair, "output_file": output_file}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from pydantic import BaseModel
import sqlite3

@app.post("/task-a10")
def task_a10(request: TaskRequest):
    """
    You are a precise data-handling LLM with expertise in database queries and financial calculations.
    Your objective is to calculate the total sales revenue from 'Gold' ticket types in a SQLite database.
    Follow these detailed steps with precision:
    1. Access the SQLite database located at '/data/ticket-sales.db'.
    2. Execute a query to calculate the total revenue (units * price) for all tickets with type 'Gold'.
    3. Retrieve the result as a numeric value. If no sales exist, treat the total as zero.
    4. Write the total sales amount as a plain number to the file '/data/ticket-sales-gold.txt'.
    5. Return the sales amount as JSON.
    Ensure proper error handling for database access issues and invalid queries.
    """
    db_file = os.path.join(DATA_DIR, "ticket-sales.db")
    output_file = os.path.join(DATA_DIR, "ticket-sales-gold.txt")

    if not os.path.exists(db_file):
        raise HTTPException(status_code=404, detail="Database file not found")

    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
        total_sales = cursor.fetchone()[0] or 0
        conn.close()

        with open(output_file, "w", encoding="utf-8") as out_file:
            out_file.write(str(total_sales))

        return {"message": "Gold ticket sales calculated successfully", "total_sales": total_sales, "output_file": output_file}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

def is_safe_path(path):
    return path.startswith(DATA_DIR) and ".." not in path

class APIRequest(BaseModel):
    api_url: str

@app.post("/phase-b/task-b3")
def fetch_api_data(request: APIRequest):
    """You are an advanced LLM designed to efficiently fetch data from external APIs, save the data into a file, and prepare it for downstream tasks. Your goal is to ensure accurate retrieval and secure storage of API data for further processing."""
    try:
        response = requests.get(request.api_url)
        if response.status_code == 200:
            save_path = os.path.join(DATA_DIR, "api_data.json")
            with open(save_path, "w") as file:
                file.write(response.text)
            return {"message": "API data saved successfully", "path": save_path}
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch API data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class GitRequest(BaseModel):
    repo_url: str
    commit_message: str

@app.post("/phase-b/task-b4")
def clone_git_repo(request: GitRequest):
    """You are a skilled LLM specializing in version control and automation. Your objective is to clone a remote git repository, create a local commit with a specified message, and ensure the repository is properly configured for future work."""
    try:
        repo_name = request.repo_url.split("/")[-1].replace(".git", "")
        repo_path = os.path.join(DATA_DIR, repo_name)

        subprocess.run(["git", "clone", request.repo_url, repo_path], check=True)
        subprocess.run(["git", "config", "--global", "user.email", "you@example.com"], check=True)
        subprocess.run(["git", "config", "--global", "user.name", "Your Name"], check=True)

        with open(os.path.join(repo_path, "new_file.txt"), "w") as file:
            file.write("This is a new file for commit.")

        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", request.commit_message], cwd=repo_path, check=True)

        return {"message": "Repository cloned and committed successfully", "repo": repo_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SQLRequest(BaseModel):
    db_path: str
    query: str

@app.post("/phase-b/task-b5")
def run_sql_query(request: SQLRequest):
    """You are a database expert LLM capable of securely executing SQL queries on SQLite databases. Your goal is to retrieve data accurately and handle database operations efficiently while ensuring data integrity."""
    try:
        conn = sqlite3.connect(request.db_path)
        cursor = conn.cursor()
        cursor.execute(request.query)
        results = cursor.fetchall()
        conn.close()
        return {"query_results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class URLRequest(BaseModel):
    url: str

@app.post("/phase-b/task-b6")
def scrape_website(request: URLRequest):
    """You are an intelligent web-scraping LLM skilled in extracting meaningful text data from websites. Your mission is to retrieve, parse, and store website content with precision, handling various HTML structures and edge cases."""
    try:
        response = requests.get(request.url)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()
        save_path = os.path.join(DATA_DIR, "scraped_text.txt")

        with open(save_path, "w") as file:
            file.write(text)

        return {"message": "Website content scraped successfully", "path": save_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ImageRequest(BaseModel):
    image_path: str
    size: int = 256

@app.post("/phase-b/task-b7")
def compress_resize_image(request: ImageRequest):
    """You are an image-processing LLM proficient in optimizing image files. Your goal is to resize and compress images while maintaining visual quality, ensuring efficient storage and faster loading times."""
    try:
        img = Image.open(request.image_path)
        img = img.resize((request.size, request.size))
        save_path = os.path.join(DATA_DIR, "compressed_image.png")
        img.save(save_path, optimize=True)

        return {"message": "Image compressed successfully", "path": save_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AudioRequest(BaseModel):
    audio_path: str

@app.post("/phase-b/task-b8")
def transcribe_audio(request: AudioRequest):
    """You are a speech-recognition LLM with advanced transcription capabilities. Your task is to convert audio recordings into accurate text transcripts, handling diverse accents, tones, and background noise."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(request.audio_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

        save_path = os.path.join(DATA_DIR, "transcription.txt")
        with open(save_path, "w") as file:
            file.write(text)

        return {"message": "Audio transcribed successfully", "path": save_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class MarkdownRequest(BaseModel):
    md_path: str

@app.post("/phase-b/task-b9")
def convert_markdown_to_html(request: MarkdownRequest):
    """You are a content-conversion LLM capable of transforming Markdown documents into clean and valid HTML. Your objective is to preserve structure and formatting during the conversion process."""
    try:
        with open(request.md_path, "r") as md_file:
            md_content = md_file.read()
            html_content = markdown.markdown(md_content)

        save_path = os.path.join(DATA_DIR, "converted.html")
        with open(save_path, "w") as html_file:
            html_file.write(html_content)

        return {"message": "Markdown converted to HTML", "path": save_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CSVRequest(BaseModel):
    file_path: str
    column: str
    value: str

@app.post("/phase-b/task-b10")
def filter_csv(request: CSVRequest):
    """You are a data-filtering LLM specializing in processing CSV files. Your role is to filter rows based on column values, returning accurate and structured JSON data suitable for analysis."""
    try:
        df = pd.read_csv(request.file_path)
        filtered_df = df[df[request.column] == request.value]
        result = filtered_df.to_dict(orient="records")

        return {"filtered_data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)