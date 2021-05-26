import selenium.webdriver.chrome.options
from selenium import webdriver
import time
import pyautogui
# last modified date: 2021-05-26

def device_setting(mode):
    # U can use other device, but need to find yourself..
    chrome_options = selenium.webdriver.chrome.options.Options()
    if mode == 'mobile':
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (iPhone; CPU iPhone OS 10_3 like Mac OS X) AppleWebKit/602.1.50 (KHTML, like Gecko) CriOS/56.0.2924.75 Mobile/14E5239e Safari/602.1')
    # type your chromeDriver location in "", include file name(chromedriver.exe)
    driver = webdriver.Chrome("D:\Chromedriver\chromedriver.exe", options=chrome_options)
    return chrome_options, driver


def geturl(url):
    # login page include return URL: if accept alert, page automatically get return url.
    returnURL = "https://login.11st.co.kr/auth/front/login.tmall?returnURL="+url
    itemURL = url
    return returnURL, itemURL
    # URL escape Code: %26: &    %2F: /    %3A: :    %3F:  ?    %3D: =


def window_size(width, height):
    driver.set_window_position(width, 0)
    driver.set_window_size(width, height)


def login(id, pw):

    driver.find_element_by_name('loginName').send_keys(id)
    driver.find_element_by_name('passWord').send_keys(pw)

    # click login button
    driver.find_element_by_class_name("btn_Atype").click()
    print('login process complete')
    # go to item page


def buying_process():
    print('wait for buying process')
    time.sleep(1)
    print('page load finished, start buying process')
    # browser.set_page_load_timeout(60)
    # try buying
    while True:
        # check class=price text
        check = driver.find_element_by_class_name("dt_price")

        if (check.text == "모바일 전용 상품입니다.\n11번가 모바일 앱에서 구매하실 수 있습니다.") or (check.text == '현재 판매중인 상품이 아닙니다.')\
                or (check.text == '일시품절로 구매가 불가합니다.'):
            driver.refresh()
            print("No Stock or not selling item")
            driver.implicitly_wait(10)

            # alert accept for message: "존재하지 않는 상품입니다."
            try:
                alert = driver.switch_to.alert
                alert.dismiss()
                alert.accept()
            except:
                print("no alert")

        else:
            buy_button = driver.find_element_by_class_name("buy")
            buy_button.click()
            time.sleep(0.3) # wait for load option list

            option_select = driver.find_element_by_class_name("select_opt")
            option_select.click()

            # full Xpath. li[number] is option number. modify u want
            option_select_phase2 = driver.find_element_by_xpath("//*[@id='optlst_0']/li[2]/a").click()
            option_select_phase2.click()

            buy = driver.find_element_by_xpath("//*[@id='optionContainer']/div[3]/div[2]/button")
            buy.click()
            break
    time.sleep(1)  # load payment page

    # use personal default buying option. default: skpay
    driver.find_element_by_class_name("btn_pay").click()

    # skpay button crawling is not possible, so input yourself(recommend) or make PW easy
    print("input SKpay password")
    time.sleep(1.2)  # load keypad

    # if click manually, delete these 6 lines, or put location ur passwd in keypad
    pyautogui.click(x=,y=)
    pyautogui.click(x=,y=)
    pyautogui.click(x=,y=)
    pyautogui.click(x=,y=)
    pyautogui.click(x=,y=)
    pyautogui.click(x=,y=)

    print("buying process finished successfully")


if __name__ == '__main__':

    # For Mobile version web site: if use desktop, device_setting("")
    mode, driver = device_setting("mobile")

    # item url: copy and paste page link. geturl('here') if use mobile version, page automatically get mobile page
    returnURL, itemURL = geturl('item_link_in_11st')
    # move to login page
    driver.get(returnURL)

    # set window size. find proper size u may see
    # window size from left top, (width, height)
    window_size(640, 1200)
    login('', '') # input id, pw

    driver.get(itemURL)
    buying_process()
