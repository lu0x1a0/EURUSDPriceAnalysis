from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

import os
chromeOptions = webdriver.ChromeOptions()


PAIR = "AUDUSD"
DOWNLOAD_FOLDER = os.getcwd()+"/Data"+PAIR
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

prefs = {"download.default_directory" : DOWNLOAD_FOLDER}
chromeOptions.add_experimental_option("prefs",prefs)

driver = webdriver.Chrome(options=chromeOptions)
driver.get("http://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/"+PAIR)


x= driver.find_elements(By.XPATH, "//div[@class='page-content']/p/a")
hrefs = [a.get_attribute('href') for a in x]

filenames = []
for href in hrefs:
    driver.get(href)
    download = driver.find_element(By.XPATH,"//a[@id='a_file']")
    filenames.append(download.text)
    download.click()

import time
while len(filenames) != 0:
    for name in filenames:
        idx = name.rfind("_")
        if os.path.exists(DOWNLOAD_FOLDER+"/"+name[:idx]+name[idx+1:]):
            print("Done: ", DOWNLOAD_FOLDER+"/"+name[:idx]+name[idx+1:])
            filenames.remove(name)
    time.sleep(5)

driver.close()
