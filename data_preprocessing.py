#Open file to preprocess
with open("fccu_hostel_policy.txt","r", encoding = "utf-8") as file: 
    data = file.readlines()

print("File opened!")

cleaned_data = []

for line in data:
    line = line.strip() #remove trailing/leading whitespace
    line = line.replace("\xa0"," ") #non breaking space characters (used to prevent line breaks. represented as "\xa0". replaced with normal space for easier preprocessing
    line = ' '.join(line.split()) #line.split() splits the line into words using whitespace as separator. returns list of words which are joined into single string
    if line: 
        cleaned_data.append(line) #empty lines excluded

print("File preprocessed!")

#Write clean data into file
with open("fccu_hostel_policy (cleaned).txt", "w", encoding = "utf-8") as file:
    for line in cleaned_data:
        file.write(line + "\n")

print("Done!")                                           

