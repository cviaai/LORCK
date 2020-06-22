import pandas as pd

REMOVAL_SHIFT_UNET = {
    "OLD":{
        "patient_21_rez":{
            "axial_pl": 2
        },
        "patient_9":{
            "axial_pl": 4,
            "axial": 8,
        },
        "patient_8":{
            "axial_pl": 6,
            "axial": 8,
        },
        "patient_17":{
            "axial": 6,
            "axial_pl": 3,
            "sagittal_pl": 1,
        },
        "patient_13":{
            "axial": 4,
            "axial_pl": 3,
        },
        "patient_14":{
            "frontal": 1,
        },
        "patient_5":{
            "axial": 4,
        },
        "patient_15":{
            "axial": 1,
        },
        "patient_19":{
            "axial": 4,
        },
    },
    "NEW":{
        "patient_15":{
            "frontal": 3
        },
        "patient_7":{
            "axial": 3,
            "axial_pl": 2,
        },
        "patient_16":{
            "sagittal_pl": 2,
        },
        "patient_5":{
            "axial": 3,
            "sagittal_pl": 3,
        },
        "patient_10":{
            "axial": 3,
            "axial_pl": 3,
        },
        "patient_11":{
            "axial": 6,
        },
        "patient_4":{
            "axial": 4,
            "frontal": 3,
        },

    }
}


REMOVAL_SHIFT_REC = {
    "OLD":{
        "patient_9":{
            "axial_pl": 100,
            "axial": 100,
        },
        "patient_8":{
            "axial_pl": 100,
            "axial": 100,
        },
        "patient_17":{
            "axial": 100,
        },
    },
    "NEW":{
        "patient_11":{
            "axial": 100,
        },

    }
}



def preprocess_dataframe_unet(df, df_split, mode='unet'):
    
    if mode == 'unet':
        REMOVAL_SHIFT = REMOVAL_SHIFT_UNET
    else:
        REMOVAL_SHIFT = REMOVAL_SHIFT_REC
    
    patients = {
        phase: {
            set_name: list(df_split[(df_split.phase == phase) 
                                    & (df_split.set == set_name)]["patients"].values)[0].split('\'')[1::2]
                    for set_name in ["OLD", "NEW"]
        } for phase in ["train", "val"]
    }
    
    index_to_rm = list(df[(df.set == "OLD") & (df.patient == "patient_2")].index) + \
                list(df[(df.set == "NEW") & (df.patient == "patient_6") & (df.seria == "axial_add")].index) + \
                list(df[(df.set == "NEW") & (df.patient == "patient_16") & (df.seria == "axial")].index) + \
                list(df[(df.set == "NEW") & (df.patient == "patient_9") & (df.seria == "axial_add")].index)
    
    for set_ in ["OLD", "NEW"]:
        for patient in REMOVAL_SHIFT[set_].keys():
            for seria in REMOVAL_SHIFT[set_][patient].keys():
                
                index_sorted = df[(df.set == set_) 
                                  & (df.patient == patient) 
                                  & (df.seria == seria)].sort_values(by="img_name").index
                
                shift = REMOVAL_SHIFT[set_][patient][seria]
                for ind in range(len(index_sorted)):
                    if ind + shift < len(index_sorted):
                        if df["if_mask"].loc[index_sorted[ind + shift]] and (~df["if_mask"].loc[index_sorted[ind]]):
                            index_to_rm += [index_sorted[ind]]
                    if ind - shift >= 0:
                        if df["if_mask"].loc[index_sorted[ind - shift]] and (~df["if_mask"].loc[index_sorted[ind]]):
                            index_to_rm += [index_sorted[ind]]
                            
    # Remove image with wrong annotation
    index_to_rm += list(df[(df.set == "OLD") 
                            &(df.patient == "patient_21_rez")
                            &(df.seria == "axial")
                            &(df.img_name.isin(["IM-0001-0015.dcm", "IM-0001-0016.dcm"]))].index) + \
                   list(df[(df.set == "OLD")
                            &(df.patient == "patient_21_rez")
                            &(df.seria == "sagittal")
                            &(df.img_name.isin(["IM-0001-0013.dcm"]))].index) + \
                   list(df[(df.set == "NEW")
                           &(df.patient == "patient_2")
                           &(df.seria == "frontal_pl")
                           &(df.img_name.isin(["IM-0001-0019.dcm"]))].index) + \
                   list(df[(df.set == "OLD") 
                           &(df.patient == "patient_5")
                           &(df.seria == "axial") 
                           &(df.img_name.isin(["IM-0001-0017.dcm"]))].index) + \
                   list(df[(df.set == "OLD") 
                           &(df.patient == "patient_6") 
                           &(df.seria == "sagittal_pl") 
                           &(df.img_name.isin(["IM-0001-0017.dcm"]))].index) + \
                   list(df[(df.set == "OLD") 
                           &(df.patient.isin(["patient_26", "patient_25"]))].index)
      
    df = df[~df.index.isin(index_to_rm)]
    
    # Dfs
    dfs = {
        phase: pd.concat( [df[(df.set == "OLD") & (df.patient.isin(patients[phase]["OLD"]))], 
                           df[(df.set == "NEW") & (df.patient.isin(patients[phase]["NEW"]))]], ignore_index=True)
        for phase in ["train", "val"]
    }

                           
                         
    return df, dfs













