# Open the text file in read mode
with open('Real_World.txt', 'r') as file:

    # Read the file's contents into a variable
    file_contents = file.read()

# Modify the contents of the file
file_contents = file_contents.replace('/DATA/disk1/hassassin/dataset/domain/OfficeHomeDataset/', '/home/chirag/OfficeHome/')

# Open the file in write mode
with open('Real_World.txt', 'w') as file:

    # Write the modified contents back to the file
    file.write(file_contents)
