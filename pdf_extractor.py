import PyPDF2
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import io
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
        return text
    except Exception as e:
        logging.warning(f"PyPDF2 failed to extract text from {pdf_path}. Falling back to OCR. Error: {str(e)}")
        return extract_text_with_ocr(pdf_path)

def extract_text_with_ocr(pdf_path):
    text = ''
    images = convert_from_path(pdf_path)
    for i, image in enumerate(images):
        text += pytesseract.image_to_string(image, lang='deu') + '\n'
    return text

def extract_tables_with_advanced_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    tables = []
    for i, image in enumerate(images):
        # Convert PIL Image to OpenCV format
        open_cv_image = np.array(image) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

        # Preprocess the image
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # Combine lines
        table_mask = horizontal_lines + vertical_lines

        # Find contours and filter for tables
        cnts = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 1000:
                x,y,w,h = cv2.boundingRect(c)
                roi = image.crop((x, y, x+w, y+h))
                ocr_result = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DATAFRAME, lang='deu')
                
                # Process OCR result to create a structured table
                table_data = []
                current_row = []
                last_top = -1
                for _, row in ocr_result.iterrows():
                    if row['conf'] != -1:  # Filter out low confidence results
                        if row['top'] != last_top and current_row:
                            table_data.append(current_row)
                            current_row = []
                        current_row.append(row['text'])
                        last_top = row['top']
                if current_row:
                    table_data.append(current_row)

                if table_data:
                    # Ensure all rows have the same number of columns
                    max_columns = max(len(row) for row in table_data)
                    padded_data = [row + [''] * (max_columns - len(row)) for row in table_data]
                    df = pd.DataFrame(padded_data[1:], columns=padded_data[0])
                    tables.append(df)

    return tables

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_date(date_str):
    date_patterns = ['%d.%m.%Y', '%d.%m.%y', '%Y-%m-%d']
    for pattern in date_patterns:
        try:
            return datetime.strptime(date_str, pattern).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return date_str

def parse_number(number_str):
    try:
        return float(number_str.replace('.', '').replace(',', '.'))
    except ValueError:
        return number_str

def extract_company_info(text):
    company_info = re.search(r'(.*?)(\d{5}\s+\w+)', text, re.DOTALL)
    if company_info:
        return {
            'name': clean_text(company_info.group(1)),
            'address': clean_text(company_info.group(2))
        }
    return {'name': '', 'address': ''}

def process_table(df):
    order_details = []
    for _, row in df.iterrows():
        if pd.notna(row.get('Dispo Nr.')) or pd.notna(row.get('Dispo Nr')):  # Handle potential variations
            order_detail = {
                'dispo_nr': str(row.get('Dispo Nr.', row.get('Dispo Nr', ''))),
                'media_type': str(row.get('Werbeträger', row.get('Werbetrager', ''))),
                'ad_format': str(row.get('Werbeform', '')),
                'date_range': {
                    'start': parse_date(str(row.get('von', ''))),
                    'end': parse_date(str(row.get('bis', '')))
                },
                'price': parse_number(str(row.get('Einzelpreis EUR', '0'))),
                'volume': parse_number(str(row.get('Volumen', '0'))),
                'unit': str(row.get('Einheit', '')),
                'gross_amount': parse_number(str(row.get('Brutto EUR', '0'))),
                'net_amount': parse_number(str(row.get('Netto EUR', '0')))
            }
            order_details.append(order_detail)
    return order_details

def extract_info_from_pdf(pdf_path):
    logging.info(f"Processing file: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    tables = extract_tables_with_advanced_ocr(pdf_path)

    company_info = extract_company_info(text)

    order_details = []
    for table in tables:
        if 'Dispo Nr.' in table.columns or 'Dispo Nr' in table.columns:
            order_details.extend(process_table(table))

    result = {
        'document_id': pdf_path,
        'document_type': 'Auftrag',
        'company_info': company_info,
        'order_details': order_details
    }

    return result

def validate_result(result):
    if not result['company_info']['name']:
        logging.warning(f"No company name found for document: {result['document_id']}")
    if not result['order_details']:
        logging.warning(f"No order details found for document: {result['document_id']}")

def main():
    pdf_files = ['auftrag1_redacted.pdf', 'auftrag2_redacted.pdf']
    results = []

    for pdf_file in pdf_files:
        try:
            result = extract_info_from_pdf(pdf_file)
            validate_result(result)
            results.append(result)
            logging.info(f"Successfully processed {pdf_file}")
        except Exception as e:
            logging.error(f"Error processing {pdf_file}: {str(e)}")

    with open('extracted_data.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logging.info("Extraction complete. Results saved to 'extracted_data.json'")

if __name__ == "__main__":
    main()