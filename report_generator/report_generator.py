# report_generator.py

import io
import os
from datetime import datetime
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.environ.get("MONGODB_URI")
if not MONGODB_URI:
    raise EnvironmentError("MONGODB_URI not set in environment variables.")
DB_NAME = "conversation_categories_db"
COLLECTION_NAME = "conversation_categories_collection"

def fetch_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    query = {
        "timestamp": {
            "$gte": start_date,
            "$lte": end_date
        }
    }

    cursor = collection.find(query)
    data = list(cursor)
    client.close()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    return df

def process_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    category_counts = df['category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']

    user_conversations = df.groupby(['user_id', 'thread_id']).size().reset_index(name='Count')

    return category_counts, user_conversations

def create_category_chart(category_counts: pd.DataFrame) -> bytes:
    plt.figure(figsize=(10, 6))
    plt.bar(category_counts['Category'], category_counts['Count'], color='skyblue')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Total Count per Conversation Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='PNG')
    plt.close()
    img_buffer.seek(0)
    return img_buffer

def create_user_conversations_chart(user_conversations: pd.DataFrame) -> bytes:
    user_totals = user_conversations.groupby('user_id')['Count'].sum().reset_index()

    plt.figure(figsize=(10, 6))
    plt.bar(user_totals['user_id'], user_totals['Count'], color='coral')
    plt.xlabel('User ID')
    plt.ylabel('Total Conversations')
    plt.title('Total Conversations per User')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='PNG')
    plt.close()
    img_buffer.seek(0)
    return img_buffer

def generate_pdf_report(start_date: datetime, end_date: datetime, output_path: str):
    df = fetch_data(start_date, end_date)
    if df.empty:
        raise ValueError("No data found for the specified time period.")

    category_counts, user_conversations = process_data(df)

    category_chart = create_category_chart(category_counts)
    user_conversations_chart = create_user_conversations_chart(user_conversations)

    doc = SimpleDocTemplate(output_path, pagesize=LETTER)
    styles = getSampleStyleSheet()
    elements = []

    title = f"Conversation Categories Report\nPeriod: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    elements.append(Paragraph(title, styles['Title']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Total Count per Conversation Category", styles['Heading2']))
    table_data = [category_counts.columns.tolist()] + category_counts.values.tolist()
    table = Table(table_data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Conversation Categories Distribution", styles['Heading2']))
    elements.append(Image(category_chart, width=400, height=300))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Total Conversations by Each User with Thread IDs", styles['Heading2']))
    table_data = [user_conversations.columns.tolist()] + user_conversations.values.tolist()
    table = Table(table_data, hAlign='LEFT', repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Total Conversations per User", styles['Heading2']))
    elements.append(Image(user_conversations_chart, width=400, height=300))
    elements.append(Spacer(1, 12))

    doc.build(elements)

    print(f"Report generated and saved to {output_path}")
