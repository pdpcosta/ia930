import selenium.common.exceptions
from selenium import webdriver
from time import sleep
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait as WDW
from selenium.webdriver.support import expected_conditions as EC


def open_music_selected_on_youtube(music_selected):
    driver = webdriver.Chrome()
    driver.get(url="https://www.youtube.com/")
    driver.implicitly_wait(10)

    search_box = WDW(driver, 10).until(EC.visibility_of_element_located((By.NAME, "search_query")))
    search_box.send_keys(music_selected)

    search_button = WDW(driver, 10).until(EC.visibility_of_element_located((By.ID, "search-icon-legacy")))
    search_button.click()

    first_video = WDW(driver, 10).until(
        EC.visibility_of_element_located((By.CLASS_NAME, "style-scope ytd-video-renderer")))
    first_video.click()

    skip_ad_button_validation(driver)

    sleep(600)
    driver.quit()


def skip_ad_button_validation(driver):
    skip_button = None
    try:
        skip_button = WDW(driver, 60).until(
            EC.presence_of_element_located((By.CLASS_NAME, "ytp-ad-skip-button-container")))
    except selenium.common.exceptions.TimeoutException:
        pass
    if skip_button is not None:
        skip_button.click()
    else:
        pass

