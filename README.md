# Problem Statement:
Clustering Indian addresses presents unique challenges and uncertainties due to the vast diversity in address formats, languages, and regional variations. The inconsistencies in the address data, presence of multiple languages, and frequent use of local landmarks make it difficult to standardize and cluster the addresses accurately. Additionally, the lack of a standardized postal address system and the prevalence of informal address components further complicate the clustering process. This project aims to develop a robust and efficient address clustering algorithm specifically tailored for Indian addresses. By accurately clustering Indian addresses, we can overcome the uncertainties and inconsistencies in the address data, and this can have various applications such as fraud detection, logistics optimization, and urban planning. Address clustering can be particularly useful in fraud detection by identifying regions with high fraudulent activities and understanding the spatial patterns of transactions, ultimately aiding in proactive measures to prevent future fraudulent cases.

# Approach:

## Data Loading

The data was read from a csv into a pandas dataframe which was later converted into a tensorflow dataset

## Data preproccessing

Tasks:

(1) Converting strings to lower case.

(2) Replacing all special characters.

(3) Removing words such as null or none.

(4) Removing extra white spaces between words. 

(5) Removing duplicates- splitting the words, finding unique words, concatenate the unique words.

(6) Eliminating short words .i.e. words of less than 20 characters in length.

## Bag of words Standardization

(1) We have an existing bag of words and we replace all similar words such as - **house no.** or **h no.** with 1 word such as **house.** 

(2) We do the same for a list of words as follows:

house,
flat,
plot,
ward,
room,
apartment,
door,
mig,
quarter,
duplex.

## Results of preprocessing and standardization

![image](https://github.com/shaun270/Address-Clustering/assets/96012817/f848aa0b-8cb3-4a2d-a1f6-5aca402b4c4e)

![image](https://github.com/shaun270/Address-Clustering/assets/96012817/f01c318b-715e-4c24-a31a-fab77207e132)



