import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    # Required parameters
    # dataset list: "wine","glass","ionosphere","muskv1","german","soybean","promoters",'wineqw',wineqr,waveform,imgseg,cnae,dr,colonTumor,DLBCLTumor,DLBCLOutcome
    # dataset bad list:magic，income，ad,tuan,
    parser.add_argument('--dataname', type=str, default = 'parkinsons',
                        choices=["glass","german","muskv1","wdbc", "colonTumor", "DLBCLStanford",
                                 "DLBCLNIH", "AMLALL","sonar","parkinsons","forestfire","promoters",
                                 "vehicle","liverdisorder","iris","movement"],
                        help='Choose absa dataset.')
    parser.add_argument('--model', type=str, default='tcdfs_k' ,
                        choices=["tcdfs_k","atcdfs_k"],
                        help='model name')
    parser.add_argument('--tcdfs_k', type=int, default=2,
                        help='Single neighborhood parameter k evaluated in this run.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for shuffled stratified folds.')
    return parser.parse_args(argv)

def dataconf():
    confdict = {}
    confdict["dataindex_reseted"] = False

    confdict["glass"] = {}
    confdict["glass"]["filepath"] = "./data/glass.data"
    confdict["glass"]["label_index"] = 10
    confdict["glass"]["seperator"] = ","
    confdict["glass"]["header"] = None
    confdict["glass"]["remove_attribute_index"] = [0] #Excluded identifier columns (when present).
    confdict["glass"]["ori_lab_condattri_idx"] = []
    confdict["glass"]["label_attributes_index"] = []

    confdict["wdbc"] = {}
    confdict["wdbc"]["filepath"] = "./data/wdbc.data"
    confdict["wdbc"]["label_index"] = 1
    confdict["wdbc"]["seperator"] = ","
    confdict["wdbc"]["header"] = None
    confdict["wdbc"]["remove_attribute_index"] = [0]  # Excluded identifier columns (when present).
    confdict["wdbc"]["ori_lab_condattri_idx"] = []
    confdict["wdbc"]["label_attributes_index"] = []

    confdict["colonTumor"] = {}
    confdict["colonTumor"]["filepath"] = "./data/colonTumor.data"
    confdict["colonTumor"]["label_index"] = 0
    confdict["colonTumor"]["seperator"] = ","
    confdict["colonTumor"]["header"] = None
    confdict["colonTumor"]["remove_attribute_index"] = []  # Excluded identifier columns (when present).
    confdict["colonTumor"]["ori_lab_condattri_idx"] = []
    confdict["colonTumor"]["label_attributes_index"] = []

    confdict["DLBCLStanford"] = {}
    confdict["DLBCLStanford"]["filepath"] = "./data/DLBCL-Stanford.data"
    confdict["DLBCLStanford"]["label_index"] = 0
    confdict["DLBCLStanford"]["seperator"] = ","
    confdict["DLBCLStanford"]["header"] = None
    confdict["DLBCLStanford"]["remove_attribute_index"] = []  # Excluded identifier columns (when present).
    confdict["DLBCLStanford"]["ori_lab_condattri_idx"] = []
    confdict["DLBCLStanford"]["label_attributes_index"] = []

    confdict["DLBCLNIH"] = {}
    confdict["DLBCLNIH"]["filepath"] = "./data/DLBCL-NIH.data"
    confdict["DLBCLNIH"]["label_index"] = 0
    confdict["DLBCLNIH"]["seperator"] = ","
    confdict["DLBCLNIH"]["header"] = None
    confdict["DLBCLNIH"]["remove_attribute_index"] = []  # Excluded identifier columns (when present).
    confdict["DLBCLNIH"]["ori_lab_condattri_idx"] = []
    confdict["DLBCLNIH"]["label_attributes_index"] = []

    confdict["AMLALL"] = {}
    confdict["AMLALL"]["filepath"] = "./data/AMLALL.data"
    confdict["AMLALL"]["label_index"] = 0
    confdict["AMLALL"]["seperator"] = ","
    confdict["AMLALL"]["header"] = None
    confdict["AMLALL"]["remove_attribute_index"] = []  # Excluded identifier columns (when present).
    confdict["AMLALL"]["ori_lab_condattri_idx"] = []
    confdict["AMLALL"]["label_attributes_index"] = []

    confdict["muskv1"] = {}
    confdict["muskv1"]["filepath"] = "./data/MuskV1.data"
    confdict["muskv1"]["label_index"] = 168
    confdict["muskv1"]["seperator"] = ","
    confdict["muskv1"]["header"] = None
    confdict["muskv1"]["remove_attribute_index"] = [0,1]#Molecule name and conformation name.
    confdict["muskv1"]["ori_lab_condattri_idx"] = []
    confdict["muskv1"]["label_attributes_index"] = []

    confdict["german"] = {}
    confdict["german"]["filepath"] = "./data/SouthGermanCredit.data"
    #confdict["german"]["label_index"] = "kredit"
    confdict["german"]["label_index"] = 20
    confdict["german"]["seperator"] = " "
    confdict["german"]["header"] = 0
    confdict["german"]["remove_attribute_index"] = []
    confdict["german"]["ori_lab_condattri_idx"] = []
    confdict["german"]["label_attributes_index"] = []



    confdict["forestfire"] = {}
    confdict["forestfire"]["filepath"] = "./data/Algerian_forest_fires_dataset_UPDATE.csv"
    confdict["forestfire"]["label_index"] = 14
    confdict["forestfire"]["seperator"] = ","
    confdict["forestfire"]["header"] = 0
    confdict["forestfire"]["remove_attribute_index"] = []
    confdict["forestfire"]["ori_lab_condattri_idx"] = [0] # Categorical region identifiers, when present.
    confdict["forestfire"]["label_attributes_index"] = [0]

    confdict["sonar"] = {}
    confdict["sonar"]["filepath"] = "./data/sonar.data"
    confdict["sonar"]["label_index"] = 60
    confdict["sonar"]["seperator"] = ","
    confdict["sonar"]["header"] = None
    confdict["sonar"]["remove_attribute_index"] = []
    confdict["sonar"]["ori_lab_condattri_idx"] = []  #
    confdict["sonar"]["label_attributes_index"] = []

    confdict["parkinsons"] = {}
    confdict["parkinsons"]["filepath"] = "./data/parkinsons.data"
    confdict["parkinsons"]["label_index"] = 17
    confdict["parkinsons"]["seperator"] = ","
    confdict["parkinsons"]["header"] = 0
    confdict["parkinsons"]["remove_attribute_index"] = [0]
    confdict["parkinsons"]["ori_lab_condattri_idx"] = []  # Categorical region identifiers, when present.
    confdict["parkinsons"]["label_attributes_index"] = []

    confdict["promoters"] = {}
    confdict["promoters"]["filepath"] = "./data/promoters.data"
    confdict["promoters"]["label_index"] = 0  # ????
    confdict["promoters"]["seperator"] = ","
    confdict["promoters"]["header"] = None
    confdict["promoters"]["remove_attribute_index"] = []
    confdict["promoters"]["ori_lab_condattri_idx"] = [i for i in range(1, 58)]
    confdict["promoters"]["label_attributes_index"] = [i for i in range(57)]

    # https://github.com/Davidgaocan/AGM_FS/
    confdict["vehicle"] = {}
    confdict["vehicle"]["filepath"] = "./data/vehicle.data"
    confdict["vehicle"]["label_index"] = 18
    confdict["vehicle"]["seperator"] = ","
    confdict["vehicle"]["header"] = None
    confdict["vehicle"]["remove_attribute_index"] = []
    confdict["vehicle"]["ori_lab_condattri_idx"] = []
    confdict["vehicle"]["label_attributes_index"] = []

    # https://github.com/Davidgaocan/AGM_FS/
    confdict["liverdisorder"] = {}
    confdict["liverdisorder"]["filepath"] = "./data/liver-disorders.data"
    confdict["liverdisorder"]["label_index"] = 6
    confdict["liverdisorder"]["seperator"] = ","
    confdict["liverdisorder"]["header"] = None
    confdict["liverdisorder"]["remove_attribute_index"] = []
    confdict["liverdisorder"]["ori_lab_condattri_idx"] = []
    confdict["liverdisorder"]["label_attributes_index"] = []

    # https://archive.ics.uci.edu/dataset/53/iris
    confdict["iris"] = {}
    confdict["iris"]["filepath"] = "./data/iris.data"
    confdict["iris"]["label_index"] = 4
    confdict["iris"]["seperator"] = ","
    confdict["iris"]["header"] = None
    confdict["iris"]["remove_attribute_index"] = []
    confdict["iris"]["ori_lab_condattri_idx"] = []
    confdict["iris"]["label_attributes_index"] = []

    # https://archive.ics.uci.edu/dataset/181/libras+movement
    confdict["movement"] = {}
    confdict["movement"]["filepath"] = "./data/movement_libras.data"
    confdict["movement"]["label_index"] = 90
    confdict["movement"]["seperator"] = ","
    confdict["movement"]["header"] = None
    confdict["movement"]["remove_attribute_index"] = []
    confdict["movement"]["ori_lab_condattri_idx"] = []
    confdict["movement"]["label_attributes_index"] = []

    return confdict

