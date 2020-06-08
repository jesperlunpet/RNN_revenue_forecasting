import csv
import operator
import xml.etree.ElementTree as et
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta

data = csv.reader(open('reports.csv'), delimiter=',')
next(data)

keyFiguresInput = ['{http://xbrl.dcca.dk/gsd}ReportingPeriodStartDate','{http://xbrl.dcca.dk/fsa}Revenue','{http://xbrl.dcca.dk/fsa}CostOfSales','{http://xbrl.dcca.dk/fsa}ExternalExpenses','{http://xbrl.dcca.dk/fsa}AdministrativeExpenses','{http://xbrl.dcca.dk/fsa}GrossResult','{http://xbrl.dcca.dk/fsa}ProfitLossFromOrdinaryOperatingActivities','{http://xbrl.dcca.dk/fsa}ProfitLossFromOrdinaryActivitiesBeforeTax','{http://xbrl.dcca.dk/fsa}ProfitLoss','{http://xbrl.dcca.dk/fsa}OtherShorttermReceivables','{http://xbrl.dcca.dk/fsa}ShorttermReceivables','{http://xbrl.dcca.dk/fsa}CashAndCashEquivalents','{http://xbrl.dcca.dk/fsa}CurrentAssets','{http://xbrl.dcca.dk/fsa}Assets','{http://xbrl.dcca.dk/fsa}ContributedCapital','{http://xbrl.dcca.dk/fsa}OtherReserves','{http://xbrl.dcca.dk/fsa}RetainedEarnings','{http://xbrl.dcca.dk/fsa}Equity','{http://xbrl.dcca.dk/fsa}OtherShorttermPayables','{http://xbrl.dcca.dk/fsa}ShorttermLiabilitiesOtherThanProvisions','{http://xbrl.dcca.dk/fsa}LiabilitiesOtherThanProvisions','{http://xbrl.dcca.dk/fsa}LiabilitiesAndEquity']

fieldnames = ['Revenue','CostOfSales','ExternalExpenses','AdministrativeExpenses','GrossResult','ProfitLossFromOrdinaryOperatingActivities','ProfitLossFromOrdinaryActivitiesBeforeTax','ProfitLoss','OtherShorttermReceivables','ShorttermReceivables','CashAndCashEquivalents','CurrentAssets','Assets','ContributedCapital','OtherReserves','RetainedEarnings','Equity','OtherShorttermPayables','ShorttermLiabilitiesOtherThanProvisions','LiabilitiesOtherThanProvisions','LiabilitiesAndEquity']

reports = {}
second = 0

for i in data:
    if (second % 2 == 0):
        reports.setdefault(i[0], set([])).add(i[1])
    second = 1+second

counter = 0

csvwriter = csv.writer(open('data.csv', 'w', newline=''), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
for key in reports:
    if (counter > 0):
        reportcontents = []
        for report in reports[key]:
            root = et.fromstring(requests.get(report).content)
            reportcontent = [None] * len(keyFiguresInput)
            for child in root:
                if child.tag in keyFiguresInput:
                    reportcontent[keyFiguresInput.index(child.tag)] = child.text
            reportcontents.append(reportcontent)
        sortedlist = sorted(reportcontents, key = lambda x: x[0])
        datechecker = iter(sortedlist)
        d1 = datetime.strptime(next(datechecker)[0], "%Y-%m-%d")
        include = True
        for report in datechecker:
            d2 = datetime.strptime(report[0], "%Y-%m-%d")
            if (relativedelta(d2, d1).years > 1):
                include = False
                break
            d1 = d2
        if include:
            csvwriter.writerow([item for sublist in sortedlist for item in sublist])
    counter = counter + 1
    if (counter % 500 == 0):
        print(counter)

# with open('test.csv', 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     for report in reports:
#         csvwriter.writerow(reduce(lambda x,y :x+y ,sorted(reports[report], key=lambda x: x[0])))    