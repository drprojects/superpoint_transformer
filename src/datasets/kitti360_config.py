import numpy as np
import os.path as osp
from collections import namedtuple
from src.datasets import IGNORE_LABEL as IGNORE


########################################################################
#                         Download information                         #
########################################################################

CVLIBS_URL = 'http://www.cvlibs.net/datasets/kitti-360/download.php'
DATA_3D_SEMANTICS_ZIP_NAME = 'data_3d_semantics.zip'
DATA_3D_SEMANTICS_TEST_ZIP_NAME = 'data_3d_semantics_test.zip'
UNZIP_NAME = 'data_3d_semantics'


########################################################################
#                              Data splits                             #
########################################################################

# These train and validation splits were extracted from:
#   - 'data_3d_semantics/2013_05_28_drive_train.txt'
#   - 'data_3d_semantics/2013_05_28_drive_val.txt'
WINDOWS = {
    'train': [
        '2013_05_28_drive_0000_sync/0000000002_0000000385',
        '2013_05_28_drive_0000_sync/0000001980_0000002295',
        '2013_05_28_drive_0000_sync/0000002282_0000002514',
        '2013_05_28_drive_0000_sync/0000002501_0000002706',
        '2013_05_28_drive_0000_sync/0000002913_0000003233',
        '2013_05_28_drive_0000_sync/0000003919_0000004105',
        '2013_05_28_drive_0000_sync/0000004093_0000004408',
        '2013_05_28_drive_0000_sync/0000004397_0000004645',
        '2013_05_28_drive_0000_sync/0000004631_0000004927',
        '2013_05_28_drive_0000_sync/0000004916_0000005264',
        '2013_05_28_drive_0000_sync/0000005249_0000005900',
        '2013_05_28_drive_0000_sync/0000005880_0000006165',
        '2013_05_28_drive_0000_sync/0000006154_0000006400',
        '2013_05_28_drive_0000_sync/0000006387_0000006634',
        '2013_05_28_drive_0000_sync/0000006623_0000006851',
        '2013_05_28_drive_0000_sync/0000006828_0000007055',
        '2013_05_28_drive_0000_sync/0000007044_0000007286',
        '2013_05_28_drive_0000_sync/0000007277_0000007447',
        '2013_05_28_drive_0000_sync/0000007438_0000007605',
        '2013_05_28_drive_0000_sync/0000007596_0000007791',
        '2013_05_28_drive_0000_sync/0000007777_0000007982',
        '2013_05_28_drive_0000_sync/0000007968_0000008291',
        '2013_05_28_drive_0000_sync/0000008278_0000008507',
        '2013_05_28_drive_0000_sync/0000008496_0000008790',
        '2013_05_28_drive_0000_sync/0000008779_0000009015',
        '2013_05_28_drive_0000_sync/0000009003_0000009677',
        '2013_05_28_drive_0000_sync/0000009666_0000009895',
        '2013_05_28_drive_0000_sync/0000009886_0000010098',
        '2013_05_28_drive_0000_sync/0000010078_0000010362',
        '2013_05_28_drive_0000_sync/0000010352_0000010588',
        '2013_05_28_drive_0000_sync/0000010577_0000010841',
        '2013_05_28_drive_0000_sync/0000010830_0000011124',
        '2013_05_28_drive_0000_sync/0000011079_0000011287',
        '2013_05_28_drive_0000_sync/0000011278_0000011467',
        '2013_05_28_drive_0002_sync/0000006383_0000006769',
        '2013_05_28_drive_0002_sync/0000006757_0000007020',
        '2013_05_28_drive_0002_sync/0000007002_0000007228',
        '2013_05_28_drive_0002_sync/0000007216_0000007502',
        '2013_05_28_drive_0002_sync/0000007489_0000007710',
        '2013_05_28_drive_0002_sync/0000007700_0000007935',
        '2013_05_28_drive_0002_sync/0000007925_0000008100',
        '2013_05_28_drive_0002_sync/0000008091_0000008324',
        '2013_05_28_drive_0002_sync/0000008311_0000008656',
        '2013_05_28_drive_0002_sync/0000008645_0000009059',
        '2013_05_28_drive_0002_sync/0000009049_0000009275',
        '2013_05_28_drive_0002_sync/0000009265_0000009515',
        '2013_05_28_drive_0002_sync/0000009502_0000009899',
        '2013_05_28_drive_0002_sync/0000009885_0000010251',
        '2013_05_28_drive_0002_sync/0000010237_0000010495',
        '2013_05_28_drive_0002_sync/0000010484_0000010836',
        '2013_05_28_drive_0002_sync/0000010819_0000011089',
        '2013_05_28_drive_0002_sync/0000011082_0000011480',
        '2013_05_28_drive_0002_sync/0000011467_0000011684',
        '2013_05_28_drive_0002_sync/0000011675_0000011894',
        '2013_05_28_drive_0002_sync/0000011885_0000012047',
        '2013_05_28_drive_0002_sync/0000012039_0000012206',
        '2013_05_28_drive_0002_sync/0000012197_0000012403',
        '2013_05_28_drive_0002_sync/0000012378_0000012617',
        '2013_05_28_drive_0002_sync/0000012607_0000012785',
        '2013_05_28_drive_0002_sync/0000012776_0000013003',
        '2013_05_28_drive_0002_sync/0000012988_0000013420',
        '2013_05_28_drive_0002_sync/0000013409_0000013661',
        '2013_05_28_drive_0002_sync/0000013652_0000013860',
        '2013_05_28_drive_0002_sync/0000013850_0000014120',
        '2013_05_28_drive_0002_sync/0000014106_0000014347',
        '2013_05_28_drive_0002_sync/0000014337_0000014499',
        '2013_05_28_drive_0002_sync/0000014491_0000014687',
        '2013_05_28_drive_0002_sync/0000014677_0000014858',
        '2013_05_28_drive_0002_sync/0000014848_0000015027',
        '2013_05_28_drive_0002_sync/0000015017_0000015199',
        '2013_05_28_drive_0002_sync/0000015399_0000015548',
        '2013_05_28_drive_0002_sync/0000015540_0000015692',
        '2013_05_28_drive_0002_sync/0000015684_0000015885',
        '2013_05_28_drive_0002_sync/0000015874_0000016223',
        '2013_05_28_drive_0003_sync/0000000274_0000000401',
        '2013_05_28_drive_0003_sync/0000000394_0000000514',
        '2013_05_28_drive_0003_sync/0000000508_0000000623',
        '2013_05_28_drive_0003_sync/0000000617_0000000738',
        '2013_05_28_drive_0003_sync/0000000731_0000000893',
        '2013_05_28_drive_0003_sync/0000000886_0000001009',
        '2013_05_28_drive_0004_sync/0000003967_0000004185',
        '2013_05_28_drive_0004_sync/0000004174_0000004380',
        '2013_05_28_drive_0004_sync/0000004919_0000005171',
        '2013_05_28_drive_0004_sync/0000005157_0000005564',
        '2013_05_28_drive_0004_sync/0000005466_0000005775',
        '2013_05_28_drive_0004_sync/0000005765_0000005945',
        '2013_05_28_drive_0004_sync/0000005930_0000006119',
        '2013_05_28_drive_0004_sync/0000006111_0000006313',
        '2013_05_28_drive_0004_sync/0000006306_0000006457',
        '2013_05_28_drive_0004_sync/0000006450_0000006647',
        '2013_05_28_drive_0004_sync/0000006637_0000006868',
        '2013_05_28_drive_0004_sync/0000006857_0000007055',
        '2013_05_28_drive_0004_sync/0000007045_0000007242',
        '2013_05_28_drive_0004_sync/0000007232_0000007463',
        '2013_05_28_drive_0004_sync/0000007449_0000007619',
        '2013_05_28_drive_0004_sync/0000007610_0000007773',
        '2013_05_28_drive_0004_sync/0000007763_0000007929',
        '2013_05_28_drive_0004_sync/0000007919_0000008113',
        '2013_05_28_drive_0004_sync/0000008103_0000008330',
        '2013_05_28_drive_0004_sync/0000008320_0000008559',
        '2013_05_28_drive_0004_sync/0000008547_0000008806',
        '2013_05_28_drive_0004_sync/0000008794_0000009042',
        '2013_05_28_drive_0004_sync/0000009026_0000009253',
        '2013_05_28_drive_0004_sync/0000009244_0000009469',
        '2013_05_28_drive_0004_sync/0000009458_0000009686',
        '2013_05_28_drive_0004_sync/0000009675_0000010020',
        '2013_05_28_drive_0004_sync/0000010156_0000010336',
        '2013_05_28_drive_0004_sync/0000010327_0000010554',
        '2013_05_28_drive_0004_sync/0000010785_0000011115',
        '2013_05_28_drive_0004_sync/0000011105_0000011325',
        '2013_05_28_drive_0004_sync/0000010010_0000010166',
        '2013_05_28_drive_0004_sync/0000010544_0000010799',
        '2013_05_28_drive_0005_sync/0000000002_0000000357',
        '2013_05_28_drive_0005_sync/0000000341_0000000592',
        '2013_05_28_drive_0005_sync/0000000579_0000000958',
        '2013_05_28_drive_0005_sync/0000000864_0000001199',
        '2013_05_28_drive_0005_sync/0000001189_0000001398',
        '2013_05_28_drive_0005_sync/0000001386_0000001669',
        '2013_05_28_drive_0005_sync/0000001653_0000001877',
        '2013_05_28_drive_0005_sync/0000001865_0000002132',
        '2013_05_28_drive_0005_sync/0000002115_0000002461',
        '2013_05_28_drive_0005_sync/0000002447_0000002823',
        '2013_05_28_drive_0005_sync/0000002807_0000003311',
        '2013_05_28_drive_0005_sync/0000003245_0000003509',
        '2013_05_28_drive_0005_sync/0000003501_0000003711',
        '2013_05_28_drive_0005_sync/0000003698_0000004017',
        '2013_05_28_drive_0005_sync/0000004007_0000004299',
        '2013_05_28_drive_0005_sync/0000004277_0000004566',
        '2013_05_28_drive_0005_sync/0000004549_0000004787',
        '2013_05_28_drive_0005_sync/0000006298_0000006541',
        '2013_05_28_drive_0006_sync/0000001208_0000001438',
        '2013_05_28_drive_0006_sync/0000001423_0000001711',
        '2013_05_28_drive_0006_sync/0000001700_0000001916',
        '2013_05_28_drive_0006_sync/0000001906_0000002133',
        '2013_05_28_drive_0006_sync/0000002124_0000002289',
        '2013_05_28_drive_0006_sync/0000002801_0000003011',
        '2013_05_28_drive_0006_sync/0000003001_0000003265',
        '2013_05_28_drive_0006_sync/0000003251_0000003634',
        '2013_05_28_drive_0006_sync/0000003613_0000003905',
        '2013_05_28_drive_0006_sync/0000003895_0000004070',
        '2013_05_28_drive_0006_sync/0000004058_0000004393',
        '2013_05_28_drive_0006_sync/0000004368_0000004735',
        '2013_05_28_drive_0006_sync/0000004723_0000004930',
        '2013_05_28_drive_0006_sync/0000004920_0000005128',
        '2013_05_28_drive_0006_sync/0000005107_0000005311',
        '2013_05_28_drive_0006_sync/0000005303_0000005811',
        '2013_05_28_drive_0006_sync/0000005801_0000005966',
        '2013_05_28_drive_0006_sync/0000005957_0000006191',
        '2013_05_28_drive_0006_sync/0000006177_0000006404',
        '2013_05_28_drive_0006_sync/0000006393_0000006648',
        '2013_05_28_drive_0006_sync/0000006639_0000006827',
        '2013_05_28_drive_0006_sync/0000006818_0000007040',
        '2013_05_28_drive_0006_sync/0000007027_0000007239',
        '2013_05_28_drive_0006_sync/0000007228_0000007465',
        '2013_05_28_drive_0006_sync/0000007457_0000007651',
        '2013_05_28_drive_0006_sync/0000007641_0000007836',
        '2013_05_28_drive_0006_sync/0000007826_0000008063',
        '2013_05_28_drive_0006_sync/0000008052_0000008284',
        '2013_05_28_drive_0006_sync/0000008271_0000008499',
        '2013_05_28_drive_0006_sync/0000008490_0000008705',
        '2013_05_28_drive_0006_sync/0000008694_0000008906',
        '2013_05_28_drive_0006_sync/0000008898_0000009046',
        '2013_05_28_drive_0006_sync/0000009038_0000009223',
        '2013_05_28_drive_0007_sync/0000000542_0000000629',
        '2013_05_28_drive_0007_sync/0000000624_0000000710',
        '2013_05_28_drive_0007_sync/0000000705_0000000790',
        '2013_05_28_drive_0007_sync/0000000785_0000000870',
        '2013_05_28_drive_0007_sync/0000000865_0000000952',
        '2013_05_28_drive_0007_sync/0000000947_0000001039',
        '2013_05_28_drive_0007_sync/0000001034_0000001127',
        '2013_05_28_drive_0007_sync/0000001122_0000001227',
        '2013_05_28_drive_0007_sync/0000001221_0000001348',
        '2013_05_28_drive_0007_sync/0000001340_0000001490',
        '2013_05_28_drive_0007_sync/0000001483_0000001582',
        '2013_05_28_drive_0007_sync/0000001577_0000001664',
        '2013_05_28_drive_0007_sync/0000001659_0000001750',
        '2013_05_28_drive_0007_sync/0000001745_0000001847',
        '2013_05_28_drive_0007_sync/0000001841_0000001957',
        '2013_05_28_drive_0007_sync/0000001950_0000002251',
        '2013_05_28_drive_0007_sync/0000002237_0000002410',
        '2013_05_28_drive_0007_sync/0000002395_0000002789',
        '2013_05_28_drive_0007_sync/0000002782_0000002902',
        '2013_05_28_drive_0009_sync/0000000002_0000000292',
        '2013_05_28_drive_0009_sync/0000000284_0000000460',
        '2013_05_28_drive_0009_sync/0000000451_0000000633',
        '2013_05_28_drive_0009_sync/0000000623_0000000787',
        '2013_05_28_drive_0009_sync/0000001385_0000001543',
        '2013_05_28_drive_0009_sync/0000001534_0000001694',
        '2013_05_28_drive_0009_sync/0000001686_0000001961',
        '2013_05_28_drive_0009_sync/0000001951_0000002126',
        '2013_05_28_drive_0009_sync/0000002117_0000002353',
        '2013_05_28_drive_0009_sync/0000002342_0000002630',
        '2013_05_28_drive_0009_sync/0000002615_0000002835',
        '2013_05_28_drive_0009_sync/0000002826_0000003034',
        '2013_05_28_drive_0009_sync/0000003026_0000003200',
        '2013_05_28_drive_0009_sync/0000003188_0000003457',
        '2013_05_28_drive_0009_sync/0000003441_0000003725',
        '2013_05_28_drive_0009_sync/0000003712_0000003987',
        '2013_05_28_drive_0009_sync/0000003972_0000004258',
        '2013_05_28_drive_0009_sync/0000004246_0000004489',
        '2013_05_28_drive_0009_sync/0000004905_0000005179',
        '2013_05_28_drive_0009_sync/0000005719_0000005993',
        '2013_05_28_drive_0009_sync/0000005976_0000006285',
        '2013_05_28_drive_0009_sync/0000006515_0000006753',
        '2013_05_28_drive_0009_sync/0000006740_0000007052',
        '2013_05_28_drive_0009_sync/0000007038_0000007278',
        '2013_05_28_drive_0009_sync/0000007264_0000007537',
        '2013_05_28_drive_0009_sync/0000007524_0000007859',
        '2013_05_28_drive_0009_sync/0000007838_0000008107',
        '2013_05_28_drive_0009_sync/0000008096_0000008413',
        '2013_05_28_drive_0009_sync/0000008391_0000008694',
        '2013_05_28_drive_0009_sync/0000008681_0000008963',
        '2013_05_28_drive_0009_sync/0000008953_0000009208',
        '2013_05_28_drive_0009_sync/0000009195_0000009502',
        '2013_05_28_drive_0009_sync/0000009489_0000009738',
        '2013_05_28_drive_0009_sync/0000009727_0000010097',
        '2013_05_28_drive_0009_sync/0000010086_0000010717',
        '2013_05_28_drive_0009_sync/0000010703_0000011118',
        '2013_05_28_drive_0009_sync/0000011099_0000011363',
        '2013_05_28_drive_0009_sync/0000011351_0000011646',
        '2013_05_28_drive_0009_sync/0000011630_0000011912',
        '2013_05_28_drive_0009_sync/0000011896_0000012181',
        '2013_05_28_drive_0009_sync/0000012167_0000012410',
        '2013_05_28_drive_0009_sync/0000012683_0000012899',
        '2013_05_28_drive_0009_sync/0000012876_0000013148',
        '2013_05_28_drive_0009_sync/0000013133_0000013380',
        '2013_05_28_drive_0010_sync/0000000002_0000000208',
        '2013_05_28_drive_0010_sync/0000000199_0000000361',
        '2013_05_28_drive_0010_sync/0000000353_0000000557',
        '2013_05_28_drive_0010_sync/0000000549_0000000726',
        '2013_05_28_drive_0010_sync/0000000718_0000000881',
        '2013_05_28_drive_0010_sync/0000000854_0000000991',
        '2013_05_28_drive_0010_sync/0000000984_0000001116',
        '2013_05_28_drive_0010_sync/0000001109_0000001252',
        '2013_05_28_drive_0010_sync/0000001245_0000001578',
        '2013_05_28_drive_0010_sync/0000001563_0000001733',
        '2013_05_28_drive_0010_sync/0000001724_0000001879',
        '2013_05_28_drive_0010_sync/0000002911_0000003114',
        '2013_05_28_drive_0010_sync/0000003106_0000003313'],

    'val': [
        '2013_05_28_drive_0000_sync/0000000372_0000000610',
        '2013_05_28_drive_0000_sync/0000000599_0000000846',
        '2013_05_28_drive_0000_sync/0000000834_0000001286',
        '2013_05_28_drive_0000_sync/0000001270_0000001549',
        '2013_05_28_drive_0000_sync/0000001537_0000001755',
        '2013_05_28_drive_0000_sync/0000001740_0000001991',
        '2013_05_28_drive_0000_sync/0000002695_0000002925',
        '2013_05_28_drive_0000_sync/0000003221_0000003475',
        '2013_05_28_drive_0000_sync/0000003463_0000003724',
        '2013_05_28_drive_0000_sync/0000003711_0000003928',
        '2013_05_28_drive_0002_sync/0000004391_0000004625',
        '2013_05_28_drive_0002_sync/0000004613_0000004846',
        '2013_05_28_drive_0002_sync/0000004835_0000005136',
        '2013_05_28_drive_0002_sync/0000005125_0000005328',
        '2013_05_28_drive_0002_sync/0000005317_0000005517',
        '2013_05_28_drive_0002_sync/0000005506_0000005858',
        '2013_05_28_drive_0002_sync/0000005847_0000006086',
        '2013_05_28_drive_0002_sync/0000006069_0000006398',
        '2013_05_28_drive_0002_sync/0000015189_0000015407',
        '2013_05_28_drive_0003_sync/0000000002_0000000282',
        '2013_05_28_drive_0004_sync/0000002897_0000003118',
        '2013_05_28_drive_0004_sync/0000003107_0000003367',
        '2013_05_28_drive_0004_sync/0000003356_0000003586',
        '2013_05_28_drive_0004_sync/0000003570_0000003975',
        '2013_05_28_drive_0004_sync/0000004370_0000004726',
        '2013_05_28_drive_0004_sync/0000004708_0000004929',
        '2013_05_28_drive_0005_sync/0000004771_0000005011',
        '2013_05_28_drive_0005_sync/0000004998_0000005335',
        '2013_05_28_drive_0005_sync/0000005324_0000005591',
        '2013_05_28_drive_0005_sync/0000005579_0000005788',
        '2013_05_28_drive_0005_sync/0000005777_0000006097',
        '2013_05_28_drive_0005_sync/0000006086_0000006307',
        '2013_05_28_drive_0006_sync/0000000002_0000000403',
        '2013_05_28_drive_0006_sync/0000000387_0000000772',
        '2013_05_28_drive_0006_sync/0000000754_0000001010',
        '2013_05_28_drive_0006_sync/0000001000_0000001219',
        '2013_05_28_drive_0006_sync/0000002280_0000002615',
        '2013_05_28_drive_0006_sync/0000002511_0000002810',
        '2013_05_28_drive_0006_sync/0000009213_0000009393',
        '2013_05_28_drive_0006_sync/0000009383_0000009570',
        '2013_05_28_drive_0007_sync/0000000002_0000000125',
        '2013_05_28_drive_0007_sync/0000000119_0000000213',
        '2013_05_28_drive_0007_sync/0000000208_0000000298',
        '2013_05_28_drive_0007_sync/0000000293_0000000383',
        '2013_05_28_drive_0007_sync/0000000378_0000000466',
        '2013_05_28_drive_0007_sync/0000000461_0000000547',
        '2013_05_28_drive_0009_sync/0000000778_0000001026',
        '2013_05_28_drive_0009_sync/0000001005_0000001244',
        '2013_05_28_drive_0009_sync/0000001234_0000001393',
        '2013_05_28_drive_0009_sync/0000004475_0000004916',
        '2013_05_28_drive_0009_sync/0000005156_0000005440',
        '2013_05_28_drive_0009_sync/0000005422_0000005732',
        '2013_05_28_drive_0009_sync/0000006272_0000006526',
        '2013_05_28_drive_0009_sync/0000012398_0000012693',
        '2013_05_28_drive_0009_sync/0000013370_0000013582',
        '2013_05_28_drive_0009_sync/0000013575_0000013709',
        '2013_05_28_drive_0009_sync/0000013701_0000013838',
        '2013_05_28_drive_0010_sync/0000001872_0000002033',
        '2013_05_28_drive_0010_sync/0000002024_0000002177',
        '2013_05_28_drive_0010_sync/0000002168_0000002765',
        '2013_05_28_drive_0010_sync/0000002756_0000002920'],

    'test': [
        '2013_05_28_drive_0008_sync/0000006988_0000007177',
        '2013_05_28_drive_0008_sync/0000000002_0000000245',
        '2013_05_28_drive_0008_sync/0000008536_0000008643',
        '2013_05_28_drive_0008_sync/0000000235_0000000608',
        '2013_05_28_drive_0008_sync/0000008417_0000008542',
        '2013_05_28_drive_0008_sync/0000004623_0000004876',
        '2013_05_28_drive_0008_sync/0000001277_0000001491',
        '2013_05_28_drive_0008_sync/0000004854_0000005104',
        '2013_05_28_drive_0008_sync/0000006792_0000006997',
        '2013_05_28_drive_0008_sync/0000002769_0000003002',
        '2013_05_28_drive_0008_sync/0000006247_0000006553',
        '2013_05_28_drive_0008_sync/0000007875_0000008100',
        '2013_05_28_drive_0008_sync/0000000812_0000001058',
        '2013_05_28_drive_0008_sync/0000007161_0000007890',
        '2013_05_28_drive_0008_sync/0000008236_0000008426',
        '2013_05_28_drive_0008_sync/0000001046_0000001295',
        '2013_05_28_drive_0008_sync/0000006517_0000006804',
        '2013_05_28_drive_0008_sync/0000005911_0000006258',
        '2013_05_28_drive_0008_sync/0000008637_0000008745',
        '2013_05_28_drive_0008_sync/0000005316_0000005605',
        '2013_05_28_drive_0008_sync/0000008090_0000008242',
        '2013_05_28_drive_0008_sync/0000005588_0000005932',
        '2013_05_28_drive_0008_sync/0000002580_0000002789',
        '2013_05_28_drive_0008_sync/0000005093_0000005329',
        '2013_05_28_drive_0008_sync/0000000581_0000000823',
        '2013_05_28_drive_0008_sync/0000002404_0000002590',
        '2013_05_28_drive_0018_sync/0000001191_0000001409',
        '2013_05_28_drive_0018_sync/0000001399_0000001587',
        '2013_05_28_drive_0018_sync/0000003503_0000003724',
        '2013_05_28_drive_0018_sync/0000002090_0000002279',
        '2013_05_28_drive_0018_sync/0000002487_0000002835',
        '2013_05_28_drive_0018_sync/0000002827_0000003047',
        '2013_05_28_drive_0018_sync/0000001577_0000001910',
        '2013_05_28_drive_0018_sync/0000000330_0000000543',
        '2013_05_28_drive_0018_sync/0000000002_0000000341',
        '2013_05_28_drive_0018_sync/0000000717_0000000985',
        '2013_05_28_drive_0018_sync/0000000530_0000000727',
        '2013_05_28_drive_0018_sync/0000000975_0000001200',
        '2013_05_28_drive_0018_sync/0000003033_0000003229',
        '2013_05_28_drive_0018_sync/0000003215_0000003513',
        '2013_05_28_drive_0018_sync/0000001878_0000002099',
        '2013_05_28_drive_0018_sync/0000002269_0000002496']}

SEQUENCES = {
    k: list(set(osp.dirname(x) for x in v)) for k, v in WINDOWS.items()}


########################################################################
#                                Labels                                #
########################################################################

# Credit: https://github.com/autonomousvision/kitti360Scripts

Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'kittiId',  # An integer ID that is associated with this label for KITTI-360
    # NOT FOR RELEASING

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'ignoreInInst',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations of instance segmentation or not

    'color',  # The color of this label
])

# A list of all labels
# NB:
#   Compared to the default KITTI360 implementation, we set all classes to be
#   ignored at train time to IGNORE. Besides, for 3D semantic segmentation, the
#   'train', 'bus', 'rider' and 'sky' classes are absent from evaluationn so we
#   adapt 'ignoreInEval', 'ignoreInInst' and 'trainId' accordingly.
#
#   See:
#   https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/evalPointLevelSemanticLabeling.py

labels = [
    #       name                     id    kittiId,    trainId   category            catId     hasInstances   ignoreInEval   ignoreInInst   color
    Label(  'unlabeled'            ,  0 ,       -1 ,    IGNORE , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,       -1 ,    IGNORE , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,       -1 ,    IGNORE , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,       -1 ,    IGNORE , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,       -1 ,    IGNORE , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,       -1 ,    IGNORE , 'void'            , 0       , False        , True         , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,       -1 ,    IGNORE , 'void'            , 0       , False        , True         , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        1 ,         0 , 'flat'            , 1       , False        , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        3 ,         1 , 'flat'            , 1       , False        , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,        2 ,    IGNORE , 'flat'            , 1       , False        , True         , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,        10,    IGNORE , 'flat'            , 1       , False        , True         , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        11,         2 , 'construction'    , 2       , True         , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        7 ,         3 , 'construction'    , 2       , False        , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        8 ,         4 , 'construction'    , 2       , False        , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,        30,    IGNORE , 'construction'    , 2       , False        , True         , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,        31,    IGNORE , 'construction'    , 2       , False        , True         , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,        32,    IGNORE , 'construction'    , 2       , False        , True         , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        21,         5 , 'object'          , 3       , True         , False        , True         , (153,153,153) ),
    Label(  'polegroup'            , 18 ,       -1 ,    IGNORE , 'object'          , 3       , False        , True         , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        23,         6 , 'object'          , 3       , True         , False        , True         , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        24,         7 , 'object'          , 3       , True         , False        , True         , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        5 ,         8 , 'nature'          , 4       , False        , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        4 ,         9 , 'nature'          , 4       , False        , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,        9 ,    IGNORE , 'sky'             , 5       , False        , True         , True         , ( 70,130,180) ),
    Label(  'person'               , 24 ,        19,        10 , 'human'           , 6       , True         , False        , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,        20,    IGNORE , 'human'           , 6       , True         , True         , True         , (255,  0,  0) ),
    Label(  'car'                  , 26 ,        13,        11 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,        14,        12 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,        34,    IGNORE , 'vehicle'         , 7       , True         , True         , True         , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,        16,    IGNORE , 'vehicle'         , 7       , True         , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,        15,    IGNORE , 'vehicle'         , 7       , True         , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,        33,    IGNORE , 'vehicle'         , 7       , True         , True         , True         , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,        17,        13 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,        18,        14 , 'vehicle'         , 7       , True         , False        , False        , (119, 11, 32) ),
    Label(  'garage'               , 34 ,        12,         2 , 'construction'    , 2       , True         , True         , True         , ( 64,128,128) ),
    Label(  'gate'                 , 35 ,        6 ,         4 , 'construction'    , 2       , False        , True         , True         , (190,153,153) ),
    Label(  'stop'                 , 36 ,        29,    IGNORE , 'construction'    , 2       , True         , True         , True         , (150,120, 90) ),
    Label(  'smallpole'            , 37 ,        22,         5 , 'object'          , 3       , True         , True         , True         , (153,153,153) ),
    Label(  'lamp'                 , 38 ,        25,    IGNORE , 'object'          , 3       , True         , True         , True         , (0,   64, 64) ),
    Label(  'trash bin'            , 39 ,        26,    IGNORE , 'object'          , 3       , True         , True         , True         , (0,  128,192) ),
    Label(  'vending machine'      , 40 ,        27,    IGNORE , 'object'          , 3       , True         , True         , True         , (128, 64,  0) ),
    Label(  'box'                  , 41 ,        28,    IGNORE , 'object'          , 3       , True         , True         , True         , (64,  64,128) ),
    Label(  'unknown construction' , 42 ,        35,    IGNORE , 'void'            , 0       , False        , True         , True         , (102,  0,  0) ),
    Label(  'unknown vehicle'      , 43 ,        36,    IGNORE , 'void'            , 0       , False        , True         , True         , ( 51,  0, 51) ),
    Label(  'unknown object'       , 44 ,        37,    IGNORE , 'void'            , 0       , False        , True         , True         , ( 32, 32, 32) ),
    Label(  'license plate'        , -1 ,        -1,        -1 , 'vehicle'         , 7       , False        , True         , True         , (  0,  0,142) ),
]

# Dictionaries for a fast lookup
NAME2LABEL = {label.name: label for label in labels}
ID2LABEL = {label.id: label for label in labels}
TRAINID2LABEL = {label.trainId: label for label in reversed(labels)}
KITTIID2LABEL = {label.kittiId: label for label in labels}  # KITTI-360 ID to cityscapes ID
CATEGORY2LABELS = {}
for label in labels:
    category = label.category
    if category in CATEGORY2LABELS:
        CATEGORY2LABELS[category].append(label)
    else:
        CATEGORY2LABELS[category] = [label]
KITTI360_NUM_CLASSES = len(TRAINID2LABEL) - 1  # 15 classes for 3D semantic segmentation
INV_OBJECT_LABEL = {k: TRAINID2LABEL[k].name for k in range(KITTI360_NUM_CLASSES)}
OBJECT_COLOR = np.asarray([TRAINID2LABEL[k].color for k in range(KITTI360_NUM_CLASSES)])
OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}
ID2TRAINID = np.array([label.trainId for label in labels])
TRAINID2ID = np.asarray([TRAINID2LABEL[c].id for c in range(KITTI360_NUM_CLASSES)] + [0])
CLASS_NAMES = [INV_OBJECT_LABEL[i] for i in range(KITTI360_NUM_CLASSES)] + ['ignored']
CLASS_COLORS = np.append(OBJECT_COLOR, np.zeros((1, 3), dtype=np.uint8), axis=0)
