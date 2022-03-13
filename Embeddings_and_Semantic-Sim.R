#################################################
# SENTENCE EMBEDDINGS AND SEMANTIC SIMILARITY #####

# Author: Marika Landau-Wells
# Version: Feb. 17, 2022 

# This script provides a toy example for preparing a text and 
# for generating sentence-level embeddings using https://huggingface.co/sentence-transformers
# models

# All code runs in R but the embeddings require a Python installation (v3) and associated
# modules (sentence-transformers) which can be installed via conda or pip

# The code requires a .txt file: "Kennan_long telegram_220246,txt" 

# If you find this example code useful, please consider citing the corresponding working paper:
# Marika Landau-Wells, "Mining Meaning: Sentence Embedding and Semantic Similarity in the Analysis of Political Text"
# (March 12, 2022)

########################################################
# Tidy up 
rm(list=ls())
gc()

# Basic inputs - additional libraries installed by section
library(foreign)
library(dplyr)

# Set as needed
setwd()

# Steps:
# 1. Import a text or corpus
# 2. Generate sentence-level dataset
# 3. Run the transformer model
# 4. Recompile the sentence-level dataset
# 5. Generate paragraph-level embeddings
# 6. Generate document-level embeddings
# 7. Calculate cosine similarity 


###################################################
## 1. Import Text or Corpus ####

library(readtext)
library(quanteda)
library(tm)
library(tokenizers)
library(stringr)
library(stringi)

# Load a document; can take advantage of readtext to add some meta-data
# Same code can be used for all .txt files in a folder to load as a single corpus
longtelegram <- readtext("Kennan_long telegram_220246.txt"
                         , docvars="filenames", docvarnames=c("Author", "DocName","Date"))

###################################################

###################################################
## 2. Generate sentence-level dataset ####

# Create a Document Index variable (useful if running many docs)
ndocs <- length(unique(longtelegram$doc_id))
longtelegram$Doc_Index <- paste(longtelegram$Author, longtelegram$Date, seq(1:ndocs), sep = "_")

# Tokenize at the paragraph-level
# This step is here to enable the creation of a paragraph-level index and to drop short paragraphs
# But it's possible to skip straight to sentences

# First check for encoding issues
longtel.doc <- stri_enc_toutf8(longtelegram$text, is_unknown_8bit = T)
longtel.paras <- tokenize_paragraphs(longtel.doc) #output is a list
nparas <- length(longtel.paras[[1]]) #77 = number of paragraphs in the document

# Unlist the paragraphs and save each within a character vector
content.vec <- c()
for (j in 1:nparas){
  content.vec[j] <- longtel.paras[[1]][j]
}

# Create an intermediate dataframe at the paragraph level
longtel.para.df <- data.frame(Author = longtelegram$Author,
                              Date = longtelegram$Date,
                              Doc_Index = longtelegram$Doc_Index,
                              Content = content.vec)
# Make sure Content is a character string
longtel.para.df$Content <- as.character(longtel.para.df$Content)
glimpse(longtel.para.df)
#longtel.para.df$Content[1:10]

# Add a Paragraph_Index variable.  
# This is useful if paragraph-level embeddings (sentence avgs) are of interest
longtel.para.df$Para_Index <- paste(longtel.para.df$Doc_Index, seq(1:nparas), sep = "_")

# At this point, it is possible to clean the text and pass the paragraphs to the embedding model
# The model demonstrated here has a 75 word cap, which suffices for most sentences, but 
# truncates many paragraphs.  Thus, more richness is preserved if the text units do not 
# Regularly exceed 75 words

# In order to preserve the paragraph-level indexing, sentences are parsed in a loop
# Here is a helper function:

sen2row <- function(x) {
  # take a dataframe generated from the filtering above and create a long df with 1 row per sentence + meta info
  n_paras <- length(x$Content)
  # get the number of paragraphs
  out.list <- list()
  # dataframe output
  for (i in 1:n_paras){
    x_row = x[i,]
    x_auth = x_row$Author
    x_date = x_row$Date
    x_doc_index = x_row$Doc_Index
    x_para_index = x_row$Para_Index
    sentences <- tokenizers::tokenize_sentences(x_row$Content)
    inter_df <- data.frame(Author = x_auth,
                           Date = x_date, 
                           Doc_Index = x_doc_index, 
                           Para_Index = x_para_index,  
                           Content = unlist(sentences))
    out.list[[i]] <- inter_df
  }
  return(out.list)
}

lt_sentences <- sen2row(longtel.para.df)

# Unlist the sentences.  Can take a while with a large corpus
system.time(
  longtel.sent.df <- as.data.frame(do.call(rbind, lt_sentences))
)
#glimpse(longtel.sent.df)
#longtel.sent.df$Content[1:10] #Check that the parsing is working
# Correct number of sentences is 249

# Return Content to a character
longtel.sent.df$Content <- as.character(longtel.sent.df$Content)

# At this point, the content can be fed into a sentence-level model
# However, some additional tidying is recommended

# Remove punctuation, numbers and extra white space using stringr and tm functions
longtel.sent.df$Content2 <- str_replace_all(longtel.sent.df$Content, "[[:punct:]]", "") %>%
  removeNumbers() %>%
  str_trim()
# Check:
#head(longtel.sent.df$Content2)
#head(longtel.sent.df$Content)

# Create a variable that counts the number of words in the string
longtel.sent.df$N_words <- str_count(longtel.sent.df$Content2, "\\w+")

# Drop extremely short texts (0-2 words)
longtel.sent.use <- filter(longtel.sent.df, N_words > 2) #Drops 13 rows
longtel.sent.use$Doc_Index <- droplevels(longtel.sent.use$Doc_Index) #Drops associated levels from factors
longtel.sent.use$Para_Index <- droplevels(longtel.sent.use$Para_Index) #Drops associated levels from factors
# Correct N is now 236 sentences

# Add a Sentence_Index variable, if desired
# Makes sense to do after filtering short stuff out
nsent_by_para <- table(longtel.sent.use$Para_Index)
# Then use a for-loop to construct a two-column dataframe (Para_Index, Sent_Index)
sent.list <- list()
for (i in 1:length(nsent_by_para)){
  maxsent <- as.numeric(nsent_by_para[i])
  sent_num <- seq(from = 1, to = maxsent)
  sent.list[[i]] <- data.frame(Para_Index = c(rep(names(nsent_by_para[i]), maxsent)),
                               Sent_Index = paste(names(nsent_by_para[i]), 
                                                  sent_num, 
                                                  sep = "_"))
}

sent.df <- as.data.frame(do.call(rbind, sent.list))
glimpse(sent.df) 

# Join sentence indices back to the full dataset
longtel.sent.use2 <- cbind(longtel.sent.use, Sent_Index = sent.df$Sent_Index)
glimpse(longtel.sent.use2)

d.longtel <- select(longtel.sent.use2, Author, Date, Doc_Index, Para_Index,
                    Sent_Index, N_words, Content2)

# Good save point
save(d.longtel, file = "Toy_df.rds") #For reference
write.csv(d.longtel, file = "Toy_df.csv", row.names = F) #For running the embeddings model 

###################################################

###################################################
## 3. Run the transformer model to get embeddings ####

library(reticulate) 

# A Python installation (3 or higher) is required.  See reticulate's documentation for
# how to set up Python for the first time
# The pip installations commands for the primary modules (packages) are: 
#!pip install transformers==3.1
#!pip3 install seaborn
# Make sure that these modules and their dependencies are stored in a single virtual environment

# Identify the python environment to use.  I use Anaconda and a virtual env called "anaconda3"
# for this project.  Replace "anaconda3" with the name of your virtual env where the necessary
# modules are installed
use_condaenv("anaconda3")

# Import the necessary modules 
py_run_string("from sentence_transformers import SentenceTransformer")
py_run_string("import pandas as pd")
py_run_string("import numpy as np")

# Import the particular model from Sentence-Transformers
# This is the model used in the paper, but a full list of available options can be found here:
# https://huggingface.co/sentence-transformers
py_run_string("model = SentenceTransformer('stsb-mpnet-base-v2')")

# Take the saved sentences .csv file ('Toy_df.csv') and create a pandas dataframe
py_run_string("df = pd.read_csv('Toy_df.csv')")

# Select only the column with the sentence text
py_run_string("messages = df['Content2'].values")

# Run the encoding model - this can take a while
py_run_string("message_embeddings = model.encode(messages)")

# The array generated will be n documents x 768 columns with no column names or row labels
# and you might need to reinitialize numpy before saving
# py_run_string("import numpy as np")

# This will put the results in your R working directory:
py_run_string("np.savetxt('LongTel_embeddings.csv', message_embeddings, delimiter=',')")



###################################################

###################################################
## 4. Recompile the sentence-level dataset ####

# The steps above preserve the document order
# Header = F will create the variable names: V1... V768
d.embeddings <- read.csv('LongTel_embeddings.csv', header = F)
head(d.embeddings)[,1:5]

# If necessary:
#load("d.longtel.rds")

# Combine the embeddings and the original data about the sentences
lt.sent.embed <- cbind(d.longtel, d.embeddings)
colnames(lt.sent.embed)[1:10]

# Drop the Content, if not using directly, to make aggregating to paragraph- and document- easier
lt.sent.embed <- select(lt.sent.embed, -Content2)
head(lt.sent.embed)[,1:10]

# Good save point
save(lt.sent.embed, file = "Toy_SentenceEmbeddings.rds")

###################################################

###################################################
## 5. Generate paragraph-level embeddings ####

# If aggregated embeddings are of interest, the following two sections provide functions for
# generating paragraph- and document-level embeddings
# If not, skip to the cosine similarity function
# Paragraphs and Documents are derived from sentence averages, consistent with other embedding
# aggregation methods

sent2para <- function(x) {
  # First calculation is the sum of all words in a paragraph
  out.x <- x %>%
    group_by(Author, Date, Doc_Index, Para_Index) %>%
    summarise_at(vars(N_words), sum, na.rm = T)
  # Second is the average of all embedding values for a given feature 
  out.v <- x %>%
    group_by(Author, Date, Doc_Index, Para_Index) %>%
    summarise_at(vars(V1:V768), mean, na.rm = T)
  out <- left_join(out.x, out.v) 
  return(out)
}

# Can take a while
system.time(
  lt.para.embed <- sent2para(lt.sent.embed)
)
head(lt.para.embed)[1:6]

# Number of paragraphs = 75
length(unique(d.longtel$Para_Index)) #also 75 

# Good save point
save(lt.para.embed, file = "Toy_ParagraphEmbeddings.rds")

###################################################

###################################################
## 5. Generate document-level embeddings ####

para2doc <- function(x) {
  # First calculation is the sum of all words in the document
  out.x <- x %>%
    group_by(Author, Date, Doc_Index) %>%
    summarise_at(vars(N_words), sum, na.rm = T) 
  out.v <- x %>%
    group_by(Author, Date, Doc_Index) %>%
    summarise_at(vars(V1:V768), mean, na.rm = T)
  out <- left_join(out.x, out.v)
  return(out)
}

lt.docs.embed <- para2doc(lt.para.embed) #only 1, of course
lt.para.embed[1, 1:6]

# Good save point
save(lt.docs.embed, file = "Toy_DocumentEmbeddings.rds")

###################################################

###################################################
## 6. Calculate cosine similarity ####

# Illustrated for sentences and paragraphs, since there is only 1 doc in the example
# Note: this is a symmetric similarity function for within a single corpus

# For the similarity function
library(text2vec)

embed2sim <- function(x, y){
  # x is a dataframe of embeddings and metadata
  # y is the Index on which the function should operate (Sent_Index, Para_Index, Doc_Index)
  # Previous functions return a grouped tibble - just make sure this is a dataframe
  x <- data.frame(x)
  # save the embeddings as df (note this is vars V1:V768)
  embed.x <- select(x, starts_with("V"))
  # save the meta-data from the embedding file, which is the first 7 cols
  meta.x <- select(x, !starts_with("V"))
  # Turn the embeddings into a matrix
  embed.mat <- data.matrix(embed.x)
  # Calculate cosine similarity within the document set (i.e., embed.mat x 2)
  sim.x <- sim2(embed.mat, embed.mat, method = "cosine")
  # Assign column names based on the level of the embeddings
  colnames(sim.x) <- meta.x[,y]
  sim.x.df <- data.frame(sim.x)
  sim.x.df <- cbind(meta.x, sim.x.df)
  # Keep the relevant index as the last column before the similarities
  sim.x.df <-  sim.x.df %>% relocate(N_words, .after = Date)
  return(sim.x.df)
}

# Generates a dataframe with sentence/paragraph/document-level meta-data appended to 
# a symmetric matrix of similarity scores for the corpus
lt.para.sim <- embed2sim(lt.para.embed, "Para_Index")
head(lt.para.sim)[1:10]

lt.sent.sim <- embed2sim(lt.sent.embed, "Sent_Index")
head(lt.sent.sim)[1:10]

