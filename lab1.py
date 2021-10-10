from nltk import PorterStemmer
from nltk.corpus import stopwords
import nltk
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
import re

nltk.download('stopwords')
# ...
from matplotlib import pyplot as plt

# sample = pd.read_csv("countries.csv")
# us = sample[sample.country == "United States"]
# china = sample[sample.country == "China"]
# plt.plot(us.year, us.population / 10**6)
# plt.plot(china.year, china.population / 10**6)
# plt.xlabel("year")
# plt.ylabel("population, mln")
# plt.legend(["us", "china"])
# plt.show()
ps = PorterStemmer()

print("Введите название файла:\n")
path = str(input())
sms = pd.read_csv(path, encoding="cp1251")

ham = sms[sms.v1 == "ham"]
spam = sms[sms.v1 == "spam"]

char = ['\d+', '\W+', '_']
ham = ham.v2.replace(char, " ", regex=True)
ham = pd.DataFrame(ham, columns=['v2'])
ham = ham['v2'].str.lower()

ham = pd.DataFrame(ham, columns=['v2'])
initHam = ham.to_numpy()
allHamWords = []
hamWordsVoc = []
hamAfterDeletingAndStemming = []
hamLength = 0
stopWords = stopwords.words('english')


for i in range(len(initHam)):
    words = str(initHam[i]).split()
    for j in range(len(words)):
        words[j] = ps.stem(words[j])

    words[0] = words[0][2:]
    words[len(words) - 1] = words[len(words) - 1][:-2]
    filtered_words = [word for word in words if word not in stopWords]

    hamAfterDeletingAndStemming.append(' '.join(filtered_words))

    for s in words:
        if s != "":
            allHamWords.append(s)
            if s not in hamWordsVoc:
                hamWordsVoc.append(s)

hamLength = [0] * len(max(hamWordsVoc, key=len))

hamCount = [0] * len(hamWordsVoc)
for s in allHamWords:
    indx = hamWordsVoc.index(s)
    hamCount[indx] += 1
    hamLength[len(s) - 1] += 1

s = 0
for n in hamLength:
    s += n * (hamLength.index(n) + 1)

hamWordLengthAvg = round(s / len(allHamWords), 2)

lengthHamMessages = 0
for m in hamAfterDeletingAndStemming:
    lengthHamMessages += len(m)
maxHamMessageLength = len(max(hamAfterDeletingAndStemming, key=len))
hamMessageAvg = round((lengthHamMessages / len(hamAfterDeletingAndStemming)), 2)

hamMessages = [0] * maxHamMessageLength
for n in hamAfterDeletingAndStemming:
    hamMessages[len(n) - 1] += 1

hamDict = {'Word': hamWordsVoc, 'Count': hamCount}
fileHam = pd.DataFrame(hamDict)
fileHam.to_csv('output/hamOutput.csv', index=False, encoding='cp1251')

ham20 = fileHam.sort_values('Count', ascending=False)
ham20 = ham20.head(20)
ham201 = ham20['Word'].tolist()
ham202 = ham20['Count'].tolist()
ham203 = []
for n in ham202:
    ham203.append(round((n / len(allHamWords)), 3))

ham1, ax1 = plt.subplots()
x1 = np.arange(1, len(max(hamWordsVoc, key=len)) + 1)
ax1.bar(x1, height=hamLength)
ax1.set_xlabel('Длина слов')
ax1.set_ylabel('Количество')
ax1.set_title('Распределение по длине слов Ham - Средняя длина' + " " + str(hamWordLengthAvg))
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
ham1.set_size_inches(15, 15)
ham1.savefig('output/ham1.png', dpi=100)

ham2, ax2 = plt.subplots()

x2 = np.arange(1, maxHamMessageLength + 1)
ax2.bar(x2, height=hamMessages)
ax2.set_xlabel('Длина уведомлений')
ax2.set_ylabel('Количество')
ax2.set_title('Распределение по длине уведомлений Ham - Средняя длина' + " " + str(hamMessageAvg))
ham2.set_size_inches(40, 10)
ham2.savefig('output/ham2.png', dpi=100)

ham3, ax3 = plt.subplots()

x3 = np.arange(20)
ax3.bar(x3, height=ham203)
ax3.set_xlabel('Слова')
ax3.set_ylabel('Частота')
ax3.set_title('Частотный анализ Ham')
ax3.set_xticks(x3)
ax3.set_xticklabels(ham201)
ham3.set_size_inches(25, 25)
ham3.savefig('output/ham3.png', dpi=100)


spam = spam.v2.replace(char, " ", regex=True)
spam = pd.DataFrame(spam, columns=['v2'])
spam = spam['v2'].str.lower()

spam = pd.DataFrame(spam, columns=['v2'])
initSpam = spam.to_numpy()
spamAfterDeletingAndStemming = []

allSpamWords = []
spamWordsVoc = []
for i in range(len(initSpam)):
    words = str(initSpam[i]).split()
    for j in range(len(words)):
        words[j] = ps.stem(words[j])

    words[0] = words[0][2:]
    words[len(words) - 1] = words[len(words) - 1][:-2]
    filtered_words = [word for word in words if word not in stopWords]

    spamAfterDeletingAndStemming.append(' '.join(filtered_words))

    for s in words:
        if s != "":
            allSpamWords.append(s)
            if s not in spamWordsVoc:
                spamWordsVoc.append(s)

spamMaxMessage = max(spamWordsVoc, key=len)
spamLeng = [0] * len(spamMaxMessage)

spamCount = [0] * len(spamWordsVoc)
for s in allSpamWords:
    indx = spamWordsVoc.index(s)
    spamCount[indx] += 1
    spamLeng[len(s) - 1] += 1

s = 0
for n in spamLeng:
    s += n * (spamLeng.index(n) + 1)

spamWordAvg = round(s / len(allSpamWords), 2)

lengthAllSpamMessages = 0
for m in spamAfterDeletingAndStemming:
    lengthAllSpamMessages += len(m)
maxSpamMessageLength = len(max(spamAfterDeletingAndStemming, key=len))
spamMessageAvg = round((lengthAllSpamMessages / len(spamAfterDeletingAndStemming)), 2)

spamMessages = [0] * maxSpamMessageLength
for n in spamAfterDeletingAndStemming:
    spamMessages[len(n) - 1] += 1



spam1, axs1 = plt.subplots()
xs1 = np.arange(1, len(spamMaxMessage) + 1)
axs1.bar(xs1, height=spamLeng)
axs1.set_xlabel('Длина слов')
axs1.set_ylabel('Количество')
axs1.set_title('Распределение по длине слов Spam - Средняя длина' + " " + str(spamWordAvg))
axs1.xaxis.set_major_locator(ticker.MultipleLocator(1))
spam1.set_size_inches(15, 15)
spam1.savefig('output/spam1.png', dpi=100)


# plt.show()

spam2, axs2 = plt.subplots()

xs2 = np.arange(1, maxSpamMessageLength + 1)
axs2.bar(xs2, height=spamMessages)
axs2.set_xlabel('Длина уведомлений')
axs2.set_ylabel('Количество')
axs2.set_title('Распределение по длине уведомлений Spam - Средняя длина' + " " + str(spamMessageAvg))
spam2.set_size_inches(40, 10)
spam2.savefig('output/spam2.png', dpi=100)


spamDict = {'Word': spamWordsVoc, 'Count': spamCount}
spamFile = pd.DataFrame(spamDict)
spamFile.to_csv('output/spamOutput.csv', index=False, encoding='cp1251')

spam20 = spamFile.sort_values('Count', ascending=False)
spam20 = spam20.head(20)
spam20_1 = spam20['Word'].tolist()
spam20_2 = spam20['Count'].tolist()
spam20_3 = []
for n in spam20_2:
    spam20_3.append(round((n / len(allSpamWords) ), 3))



spam3, axs3 = plt.subplots()
xs3 = np.arange(20)
axs3.bar(xs3, height=spam20_3)
axs3.set_xlabel('Слова')
axs3.set_ylabel('Частота')
axs3.set_title('Частотный анализ Spam')
axs3.set_xticks(xs3)
axs3.set_xticklabels(spam20_1)
spam3.set_size_inches(25, 25)
spam3.savefig('output/spam3.png', dpi=100)

print("Введите сообщение для анализа:")
inputMessage = str(input())
inputMessage = re.sub(r'[^\w\s]+|[\d]+', r'', inputMessage).strip()

inputMessageWords = inputMessage.split()
allSpamWordsCountc = len(allSpamWords)
allHamWordsCountc = len(allHamWords)

filtered_words = [word for word in inputMessageWords if word not in stopWords]

inputMessageWords = filtered_words

isSpam = len(spam) / len(sms)
for s in inputMessageWords:
    if s not in spamWordsVoc:
        allSpamWordsCountc += 1

for s in inputMessageWords:
    if s in spamWordsVoc:
        i = spamWordsVoc.index(s)
        isSpam *= spamCount[i] / allSpamWordsCountc
    else:
        isSpam *= (1 / allSpamWordsCountc)

isHam = len(ham) / len(sms)
for s in inputMessageWords:
    if s not in hamWordsVoc:
        allHamWordsCountc += 1

for s in inputMessageWords:
    if s in hamWordsVoc:
        i = hamWordsVoc.index(s)
        isHam *= hamCount[i] / allHamWordsCountc
    else:
        isHam *= 1 / allHamWordsCountc

print("Вероятность, что сообщение - Spam:")
print(isSpam)
print("Вероятность, что сообщение - Ham:")
print(isHam)

print("Cкорее всего, что сообщение - Spam") if isSpam > isHam else print("Cкорее всего, что сообщение - Ham")
