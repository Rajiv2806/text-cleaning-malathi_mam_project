rm(list = ls())

library(text2vec)
library(data.table)
library(stringr)
library(tm)
library(RWeka)
library(tokenizers)
library(slam)
library(textir)
library(wordcloud)

text.clean = function(x)
{ require("tm")
    x  =  gsub("<.*?>", " ", x)
    x  =  iconv(x, "latin1", "ASCII", sub="")
    x  =  gsub("[^[:alnum:]]", " ", x)
    x  =  tolower(x)
    x  =  removeNumbers(x)
    x  =  stripWhitespace(x)
    x  =  gsub("^\\s+|\\s+$", "", x)
    return(x)
}


input_file1 = readLines(file.choose()) #, header = TRUE)  #colnames(input_file)  #dim(input_file)
data = data.frame(Id = 1:length(input_file1), Original_Text = input_file1, stringsAsFactors = F)
data = data[2:nrow(data),1:ncol(data)]

stpw1 = readLines('https://raw.githubusercontent.com/sudhir-voleti/basic-text-analysis-shinyapp/master/data/stopwords.txt')
stpw2 = tm::stopwords('english')
stpw3 = c('gg','ad','hd','ag','rs','mp','ap','ms','pdf','iqn','from','to','cc','pm','am','subject','gigigarcia'
          ,'re','fw','hd','gg','ca','volt','email','acct','inv','www','emailed','vmc','coll','tel','ck','rk')
name = c('hema','gonzalez','mailto','dadlani','gigi','garcia','lesley','modesta','glassell','anna','irshad','ahmed'
         ,'perez','paula','durling','boerner','terry','mcdougall','sanmina','cruz','vroman','germaine','rashmi'
         ,'denise','shepherd','kaveramma','jordan','ranjan')
month = c('january','feburary','march','april','may','june','july','august','september','october'
          ,'november','december','jan','feb','mar','apr','may','jun','jul','aug','sept','oct','nov','dec')
week = c('monday','tuesday','wednesday','thursday','friday','saturday','sunday'
         ,'mon','tues','wed','thurs','fri','sat','sun','week','date','dated','hour','hours',
         'thank','you','good','morning','evening','afternoon')
comn  = unique(c(stpw1, stpw2, stpw3,name,month,week))
stopwords = unique(gsub("'"," ",comn)) 

data$Cleaned_Text = data$Original_Text  
data$Cleaned_Text = text.clean(data$Cleaned_Text) #head(data,2)
data$Cleaned_Text = removeWords(data$Cleaned_Text,stopwords)
data$Cleaned_Text = stripWhitespace(data$Cleaned_Text)

data = data[nchar(data$Cleaned_Text)>0,1:3]
data$Id = c(1:nrow(data)) #head(data)

write.csv(data,"Cleaned Notes.csv",row.names = F)
getwd()


## STOP HERE FOR JUST DATA CLEANING
######################################################################################################
## BELOW IS BASIC TEXT ANALYTICS PERFORMED - TF & TFIDF 

data$Cleaned_Text = gsub("invoices","invoice",data$Cleaned_Text)
data$Cleaned_Text = gsub("pmt","payment",data$Cleaned_Text)
data$Cleaned_Text = gsub("currnt","current",data$Cleaned_Text)


#x = data$Cleaned_Text #y = as.String(data$Cleaned_Text)
#y = x #library(qdap) #z = stemmer(y)
#z = y #head(z) #(unique(word_tokenizer(y)))
#library(wordcloud) #wordcloud(z,max.words = 50)

it_m = itoken(data$Cleaned_Text,tokenizer = word_tokenizer,ids = 1:length(data$Cleaned_Text),progressbar = T)
vocab = create_vocabulary(it_m,ngram = c(2L,2L))
pruned_vocab = prune_vocabulary(vocab,term_count_min = 20)
vectorizer = vocab_vectorizer(pruned_vocab)
dtm_m  = create_dtm(it_m, vectorizer);dim(dtm_m) #(dtm_m[1:5,1:5]) 
tsum = as.matrix(t(rollup(dtm_m, 1, na.rm=TRUE, FUN = sum)))
tsum = tsum[order(tsum, decreasing = T),]   #head(tsum,50);tail(tsum)

for (term in names(tsum)){
    focal.term = gsub("_", " ",term)
    replacement.term = term
    data$Cleaned_Text = gsub(paste("",focal.term,""),paste("",replacement.term,""), data$Cleaned_Text)
}

it_m = itoken(data$Cleaned_Text,tokenizer = word_tokenizer,ids = 1:length(data$Cleaned_Text),progressbar = T)
vocab = create_vocabulary(it_m)
pruned_vocab = prune_vocabulary(vocab,term_count_min = 20)
vectorizer = vocab_vectorizer(pruned_vocab)
dtm_m  = create_dtm(it_m, vectorizer);dim(dtm_m) #(dtm_m[1:5,1:5]) 
tsum1 = as.matrix(t(rollup(dtm_m, 1, na.rm=TRUE, FUN = sum)))
tsum1 = tsum1[order(tsum1, decreasing = T),]   #head(tsum1,50);tail(tsum1)

windows()
wordcloud(names(tsum1), tsum1,scale = c(4, 0.5),1,max.words = 100,colors = brewer.pal(8, "Dark2"))
title(sub = "Term Frequency - Wordcloud")  

dtm = as.DocumentTermMatrix(dtm_m, weighting = weightTf);
a0 = (apply(dtm, 1, sum) > 0)
dtm = dtm[a0,]           

dtm.tfidf = tfidf(dtm, normalize=TRUE)

tst = round(ncol(dtm.tfidf)/100)
a = rep(tst, 99)
b = cumsum(a);rm(a)
b = c(0,b,ncol(dtm.tfidf))

ss.col = c(NULL)
for (i in 1:(length(b)-1)) {
    tempdtm = dtm.tfidf[,(b[i]+1):(b[i+1])]
    s = colSums(as.matrix(tempdtm))
    ss.col = c(ss.col,s) }
tsum2 = ss.col
tsum2 = tsum2[order(tsum2, decreasing = T)]; #head(tsum2,50)

windows()
wordcloud(names(tsum2),tsum2,scale = c(4,.5),colors = brewer.pal(8,"Dark2"),max.words = 100)
title(sub = "Term Frequency Inverse Document Frequency - Wordcloud")  
