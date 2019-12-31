from google_images_download import google_images_download
import argparse

list_keyword = [
    'debat 2019','jokowi', 'ma ruf amin','prabowo subianto' ,'sandiaga uno']

for keyword in list_keyword:
    print("~~~ START ", keyword)

    response = google_images_download.googleimagesdownload()
    arguments = {"keywords":keyword,"print_urls":True}
    paths = response.download(arguments)   #passing the arguments to the function
    print("~~~ DONE ", keyword)
