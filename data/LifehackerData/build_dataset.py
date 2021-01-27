import urllib
from bs4 import BeautifulSoup
import time
import random

def spider(name, found_titles, url, found):
    try:
        page = urllib.request.urlopen(url)
        soup = BeautifulSoup(page, 'html.parser')
        # getting the description of the article
        title = soup.select('meta[name="description"]')[0]['content']
        if len(title) <= 400:
            # getting keywords
            keywords = soup.select('meta[name="keywords"]')[0]['content'].lower().split(', ')

            # Removing the word 'lifehacker' from the keywords, bc it is in every article.
            if name in keywords:
                keywords.remove(name)

            cleaned_keywords = []

            # checking if all the keywords are actually in description
            for k in keywords:
                if k in title.lower():
                    cleaned_keywords.append(k)

            # writing everything to a file
            if len(cleaned_keywords) > 0 and title not in found_titles:
                found_titles.append(title)
                print(title)
                print(cleaned_keywords)

                f = open('lifehacker-data2.txt', 'a')
                f.write(
                    title + "\t" + ' '.join(
                        k.replace(' ', '_') for k in cleaned_keywords
                    ) + "\n"
                )
                f.close()

        # Here the problems begin:
        # collecting all the links on the page to continue webscraping
        # the other lifehacker articles as well
        for a in soup.select('a[href]'):
            b = a['href'].replace('#replies', '')
            if 'https://' + name + '.com' in b and b not in found:

                found.append(b)
                # calling the function again
                t = random.uniform(1, 20)
                time.sleep(t)
                spider(name, found_titles, b, found)

    except:
        print('Exception occured')
        pass

def main():
    name      = 'lifehacker'
    start_url = 'https://offspring.lifehacker.com/how-to-talk-to-kids-about-the-holocaust-1846134591'
    spider(name, [], start_url, [start_url])

if __name__ == "__main__":
    main()