#fitz: library to handle/extract text from pdfs. part of PyMuPDF library


import fitz

def pdf_data_extract(file_name):
    pdf = fitz.open(file_name)
    data = ""

    for page in range(len(pdf)): #len(pdf) will return total number of pages in pdf
        page_content = pdf[page]
        data+=page_content.get_text() #get_text() to extract data from page

    pdf.close()
    return data


#ensure correct directory provided
#raw string literals: where \ is treated as normal character
#add r for raw string literal to ensure \ is not treated as escape sequences
pdf_name = r"D:\FCCU Admission Assistant (Final)\Hostel-Policy-61119.pdf"
pdf_data = pdf_data_extract(pdf_name)
print(pdf_data)

#utf.8: character encoding that can handle/store wide range of characters correctly e.g. symbols, non english words
with open("fccu_hostel_policy.txt","w", encoding = "utf-8") as file:
    file.write(pdf_data)

print("Text Extraction Successful")



