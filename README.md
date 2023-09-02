# Problem Statement:
Clustering Indian addresses presents unique challenges and uncertainties due to the vast diversity in address formats, languages, and regional variations. The inconsistencies in the address data, presence of multiple languages, and frequent use of local landmarks make it difficult to standardize and cluster the addresses accurately. Additionally, the lack of a standardized postal address system and the prevalence of informal address components further complicate the clustering process. This project aims to develop a robust and efficient address clustering algorithm specifically tailored for Indian addresses. By accurately clustering Indian addresses, we can overcome the uncertainties and inconsistencies in the address data, and this can have various applications such as fraud detection, logistics optimization, and urban planning. Address clustering can be particularly useful in fraud detection by identifying regions with high fraudulent activities and understanding the spatial patterns of transactions, ultimately aiding in proactive measures to prevent future fraudulent cases.

# Approach:

The following methods is the flow according to the Address_Clustering.ipynb file.

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

One drawback that I face now is that sorting the tensorflow dataset on the basis of pincode is not possible, since tensorflow datasets are not meant for that.

Hence, I converted it back to pandas to continue with the remaining process.

Please refer to the bge.ipynb file for further results.

The final result of preprocessing and standardization is stored in output.csv

## Reading the data

The data is stored in a pandas dataframe.

## Sorting on basis of pincode and ascii value

The methodology behind this step is driven by the reason that 

(1) Comparisons are done within pincodes (N X N).

(2) To optimize, the strings of nearly equal lengths will be compared as a 1:1 comparison .i.e.

address0 compared with address1

address1 compared with address2

and so on..

## Loading the BGE embeddings model and creating embeddings

As on now that is September 2023 the BGE embedding model is the first on the MTEB leaderboard https://huggingface.co/spaces/mteb/leaderboard even surpassing OpenAi embeddings.

The BGE from hugging face was used to encode the 'Delivery_Desc' column into vectors which would later be used for clustering.

Here is the link for the model https://huggingface.co/BAAI/bge-large-en

## Clustering

### DBSCAN method

DBSCAN, which stands for Density-Based Spatial Clustering of Applications with Noise, is a popular clustering method that is fundamentally different from partitioning methods like k-means. Instead of classifying each point into a cluster and adjusting the clusters iteratively, DBSCAN groups together points that are close to each other based on a distance measurement (usually Euclidean distance but for my purpose I have used cosine similarity ) and a minimum number of points. It also marks as outliers the points that are in low-density regions.

DBSCAN works by defining a neighborhood around each data point, and then growing clusters by connecting neighborhoods of neighboring points. More specifically, for each point in the dataset, DBSCAN gathers all points that are at a maximum distance 'eps' from it, and if this neighborhood contains at least 'min_samples' points, a new cluster is created, and the algorithm continues by exploring the neighbors' neighbors and so on. If a neighboring point is already part of another cluster, the two clusters are connected. If a point cannot be added to any cluster (because its neighborhood does not contain enough points), it is marked as noise.

The advantages of DBSCAN are that it does not assume any inherent structure for the data, does not require the number of clusters to be specified beforehand, and can find arbitrarily shaped clusters. It is particularly well-suited for applications where there may be clusters of similar density but different shapes. However, it can struggle with clusters of varying densities and is sensitive to the settings of 'eps' and 'min_samples'.

The results it provided were unsatisfactory as visible in the bge.ipynb file.

### 1:1 comparisons

This step-by-step comparison minimizes the computational load by avoiding unnecessary comparisons, and also takes advantage of the fact that addresses with similar lengths are more likely to be similar or related, making the clustering process more efficient and effective.

This gave somewhat satisfactory results but however still couldnt get all the desired addresses.

### less than N:N comparisons within pincode

This method efficiently clusters addresses by performing selective comparisons rather than exhaustive N:N comparisons. The algorithm identifies unique pincodes, initializes clusters, and compares address embeddings within the same pincode using cosine similarity. If the cosine similarity is greater than 0.85, the address is added to the existing cluster; otherwise, a new cluster is created. Finally, clusters with less than two addresses are removed. This approach optimizes computational resources while maintaining a reasonable level of clustering accuracy.

The results this method gave were much better but however there is still some scope for improvement.





