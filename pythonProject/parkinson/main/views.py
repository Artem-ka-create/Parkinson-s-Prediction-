import json
import sys

from sklearn.tree import DecisionTreeClassifier
sys.path.append('main/')

from spiralupload import *
from microupload import *
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, make_scorer, fbeta_score
from sklearn.model_selection import train_test_split, KFold
from .models import spiral_list_Models
from .models import micro_list_Models
from .forms import PatientForm
from django.shortcuts import render
from .forms import ImageForm
from .forms import DocumentForm
from .models import Image
import os
from pathlib import Path
import cv2
import numpy as np
from imutils import paths
from skimage import feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def handle_uploaded_file(f):
    with open('some/file/name.txt', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


# RandomForestClassifier model
def RandomForest_model(X_train,y_train):

    grid_params = {
        'n_estimators': [5, 2,7],'max_features': ['auto', 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],'max_depth': [1, 2, 4]
    }

    ftwo_scorer = make_scorer(fbeta_score, beta=1)
    model = GridSearchCV(
        RandomForestClassifier(random_state=57),
        grid_params,cv=9,verbose=True,
        n_jobs=-1,return_train_score=True,scoring=ftwo_scorer)

    return model.fit(X_train, y_train)

# DecisionTreeClassifier model
def DTclass_model(X_train,y_train):
    tree_para = {'criterion': ['gini', 'entropy'],
                 'max_features': ['auto', 'sqrt', 'log2'],
                 'ccp_alpha': [0.1, .01, .001],
                 'max_depth': [1, 2, 3, 4]}

    ftwo_scorer = make_scorer(fbeta_score, beta=1)
    model = GridSearchCV(DecisionTreeClassifier(random_state=1024),
                      tree_para,cv=14,return_train_score=True,
                      refit=True,n_jobs=-1,verbose=8,
                      scoring=ftwo_scorer)

    return model.fit(X_train,y_train)

# SupportVectorClassifier model
def SVClass_model(X_train,y_train):

    svc = SVC(kernel='sigmoid', random_state=274)
    cv = KFold(n_splits=23, shuffle=True, random_state=274)
    grid = {'C': np.power(10.0, np.arange(-1, 1.5)),
        'kernel': ['rbf', 'poly']}
    model = GridSearchCV(svc, grid, refit=True, cv=cv,
                      verbose=True, return_train_score=True)


    return model.fit(X_train,y_train)

def get_statistics(y_test, predictions):

    cm = confusion_matrix(y_test, predictions).flatten()
    (tn, fp, fn, tp) = cm
    print("CMatrix= ", cm)
    accuracy_score = (tp + tn) / float(cm.sum())
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_measure = (2 * recall * precision) / (recall + precision)
    print("acc - ", accuracy_score)
    print("prec - ", precision)
    print("recall - ", recall)
    print("f1 - ", f1_measure)

    return accuracy_score

def get_neural_prediction(modelChoose,MDVPFhi,MDVPFlo,MDVPJitterProcent,
                        MDVPJitterABS,MDVPRAP,MDVPPPQ,JitterDDP,MDVPShimmer,
                        MDVPShimmerdB,ShimmerAPQ3,ShimmerAPQ5,MDVDFo,MDVPAPQ,
                        ShimmerDDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE):

    #upload of dataset
    data = pd.read_csv('main/parkinsons.data')
    data.head()

    #preprocessing by MinMaxScaler
    scaler = MinMaxScaler((-1, 1))

    # split to 'x' and 'y' datas
    x = (scaler.fit_transform(data.drop(columns=['name', 'status'], axis=1)))
    y = data['status']

    #check of model names
    if (modelChoose == 'SupportVectorClassifier'):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=10)
        model = SVClass_model(X_train, y_train)

    if (modelChoose == 'DecisionTreeClassifier'):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
        model = DTclass_model(X_train, y_train)

    if(modelChoose=='RandomForestClassifier'):
        X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.3, random_state=7)
        model = RandomForest_model(X_train, y_train)

    #getting predictions by testing datas
    predictions = model.predict(X_test)

    accuracy_score=get_statistics(y_test,predictions)

    #creating a vector of user's data
    input_date = (MDVDFo, MDVPFhi, MDVPFlo, MDVPJitterProcent, MDVPJitterABS, MDVPRAP, MDVPPPQ, JitterDDP,
                  MDVPShimmer, MDVPShimmerdB, ShimmerAPQ3, ShimmerAPQ5, MDVPAPQ, ShimmerDDA, NHR, HNR,
                  RPDE, DFA, spread1, spread2, D2, PPE)

    input_date_array = np.asarray(input_date)
    input_date_reshape = input_date_array.reshape(1, -1)
    stddata = scaler.transform(input_date_reshape)

    #get prediction by user's data
    result = model.predict(stddata)

    return result[0],accuracy_score

# SupportVectorClassifier model
def SVC_model(trainX,trainY):
    # svc = SVC()
    # param_grid = {'C': [0.1, 1, 10, 100],
    #               'gamma': [1, 0.1, 0.01, 0.001],
    #               'kernel': ['poly', 'sigmoid']}
    # model = GridSearchCV(svc, param_grid, refit=True, verbose=2)

    svc = SVC(kernel='sigmoid', random_state=254)
    cv = KFold(n_splits=12, shuffle=True, random_state=254)
    grid = {
        'C': np.power(10.0, np.arange(-1, 1)),
        'kernel': ['rbf', 'poly', 'sigmoid'],
    }
    model = GridSearchCV(svc, grid, refit=True, cv=cv,
                      verbose=9, return_train_score=True)

    return model.fit(trainX, trainY)

# XGBoost Classifier model
def XGBooST(trainX,trainY):

    cv = KFold(n_splits=4, shuffle=True, random_state=254)

    estimator = XGBClassifier()
    parameters = {
        # 'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              # 'learning_rate': [1, 0.1], #so called `eta` value
              'max_depth': range(1, 30, 10),
              'min_child_weight': [11],
              # 'silent': [1],
              # 'subsample': [0.8],
              'colsample_bytree': [0.75],
              'n_estimators': range(11, 55, 33), #number of trees, change it to 1000 for better results
              # 'missing':[-999],
              'seed': [12]
    }
    model = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        n_jobs=-1,
        cv=cv,
        scoring='accuracy',
        return_train_score=True,
        verbose=1,
        refit=True
    )


    return model.fit(trainX, trainY)

# RandomForest Classifier model
def RandomForest(trainX,trainY):
    grid_params = {
        'n_estimators': [5, 10, 50],
        'max_features': ['auto', 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
        'max_depth': [1, 2, 4]
    }
    ftwo_scorer = make_scorer(fbeta_score, beta=1)
    model = GridSearchCV( RandomForestClassifier(random_state=57),
        grid_params,cv=9, verbose=7, n_jobs=-1,
        return_train_score=True, scoring=ftwo_scorer)

    return model.fit(trainX, trainY)

def preprocess_image(imagePath):

    # preprocessing of image by making
    # more readable for hog.function
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (200, 200))

	image = cv2.threshold(image, 0, 255,
						  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

	return image
def quantify_image(image):

    #create transport picture to HOG and get vector
    vector = feature.hog(image, orientations=9,
        pixels_per_cell=(10, 10), cells_per_block=(2, 2),
        transform_sqrt=True, block_norm="L1")

    # return vector
    return vector
def to_vector(path):

    # get an array of all images paths
    images = list(paths.list_images(path))
    # creating of saving arrays
    vector = []
    labels = []
    # loop over the image paths
    for image in images:
        label = image.split(os.path.sep)[-2]
        # open-cv fuctions
        im = preprocess_image(image)
        # from image to vector of numbers
        features = quantify_image(im)
        # add new feature and label
        vector.append(features)
        labels.append(label)

    result=np.array(vector), np.array(labels)
    # return the data and labels
    return result

def picture_prediction(modelChoose):

    args = dict()
    #get paths to dataset
    if (modelChoose=="SupportVectorClassifier"):
        args["dataset"] = 'main/spiral_dataset/datasetSVC/spiral'

    elif (modelChoose == "RandomForestClassifier"):
        args["dataset"] = 'main/spiral_dataset/datasetLogRF/spiral'
    else:
        args["dataset"] = 'main/spiral_dataset/datasetXGBOOST/spiral'

    # get training and testing dataset paths
    trainingPath = os.path.sep.join([args["dataset"], "training"])
    testingPath = os.path.sep.join([args["dataset"], "testing"])
    #save of picture path
    resultPath = 'media/images'


    # from pictures to numb vector tranformation
    # preprocessing of data
    (trainX, trainY) = to_vector(trainingPath)
    (testX, testY) = to_vector(testingPath)
    le = LabelEncoder()
    trainY = le.fit_transform(trainY)
    testY = le.transform(testY)


    # choose of model by clicked block on web-aplication
    if(modelChoose=="XGBClassifier"):
        model = XGBooST(trainX,trainY)

    if(modelChoose=="SupportVectorClassifier"):
        model = SVC_model(trainX,trainY)

    if(modelChoose=="RandomForestClassifier"):
        model = RandomForest(trainX,trainY)

    # getting of prediction values from testing
    predictions = model.predict(testX)

    # getting of statistics
    accuracy_score=get_statistics(testY,predictions)

    # getting path to user's picture
    testingPaths = list(paths.list_images(resultPath))


    # preprocess and get prediction by user's picture
    image = cv2.imread(testingPaths[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = quantify_image(image)
    preds = model.predict([features])
    print(preds)

    return preds[0],accuracy_score

def delete_from_folder(path,type):
    type='*.'+type
    for f in Path(path).glob(type):
        try:
            f.unlink()
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))
def index(request):
    return render(request, 'main/index.html')
def about(request):
    return render(request, 'main/about.html')
def support(request):
    return render(request, 'main/support.html')
def clear_media_folder():
    type = '*'
    for f in Path('media/images').glob(type):
        f.unlink()
    for f in Path('media/files').glob(type):
        f.unlink()


# delete all user's models and download default
def get_default_models_spiral():

    default_spir_models = ['XGBClassifier', 'SupportVectorClassifier', 'RandomForestClassifier']

    for i in range(0, len(default_spir_models)):
        models = spiral_list_Models(spiral_list='', spiral_model_name=default_spir_models[i])
        models.save()
# delete all user's models and download default
def get_default_models_micro():
    default_micro_models = ["DecisionTreeClassifier", "RandomForestClassifier", "SupportVectorClassifier"]

    for i in range(0, len(default_micro_models)):
        models = micro_list_Models(spiral_list='', spiral_model_name=default_micro_models[i])
        models.save()

# script of to deafault batton
def to_default(request):
    clear_media_folder()

    spiral_list_Models.objects.all().delete()
    micro_list_Models.objects.all().delete()

    get_default_models_spiral()
    get_default_models_micro()


    modelmicro = micro_list_Models.objects.all()
    modelspir = spiral_list_Models.objects.all()
    print(modelspir)
    print(modelmicro)

    return render(request, "main/uploadModel.html")

# update code of upload model
def update_upl_model(path_main,path_upload):

    main = open(path_upload, 'r')
    model = main.read()

    print(path_upload)
    print(model)
    main.close()

    f = open(path_main, 'w')
    f.write(model)
    f.close()
def uploadModel(request):
    # check of method request
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)

        # get name of model from url
        req = json.dumps(request.POST)
        print(req)
        method = req.split(':')[-1].replace('}', '').replace(' ', '').replace('"', '')
        nameOfModel = req.split(':')[-2].replace('}', '').replace(' ', '').replace('"', '')
        nameOfModel = nameOfModel.split(',')[0]


        try:
            # check on right of uploaded form
            if form.is_valid():
                # check of method getting information
                # it is important if you get NEW models
                if method == 'micro' :
                    print(nameOfModel)
                    if len(micro_list_Models.objects.all()) < 4:
                            print("trhy")
                        # try:
                            models = micro_list_Models(spiral_list='', spiral_model_name=nameOfModel)
                            models.save()
                            form.save()
                            message = 'Uploaded'
                            error = ''

                            parent = 'media/files'


                            path_upload = os.path.join(parent, nameOfModel + '.py')
                            path_main = 'main/microupload.py'

                            update_upl_model(path_main, path_upload)
                        # except Exception:
                        #     print(Exception)
                        #     to_default(request)


                    else:

                        delmodel = list(micro_list_Models.objects.values_list('spiral_model_name', flat=True))[-1]
                        print("eeell")
                        micro_list_Models.objects.filter(spiral_model_name=delmodel).delete()

                        models = micro_list_Models(spiral_list='', spiral_model_name=nameOfModel)

                        print(models)
                        models.save()
                        form.save()

                        parent = 'media/files'
                        path = os.path.join(parent, delmodel + '.py')
                        # print(path)

                        os.remove(path)

                        message = 'Uploaded another model'
                        error = ''

                        parent = 'media/files'
                        path_upload = os.path.join(parent, nameOfModel + '.py')
                        path_main = 'main/microupload.py'

                        update_upl_model(path_main, path_upload)
                        print("!!!a")


                elif method == 'spiral':
                    print(nameOfModel)
                    if len(spiral_list_Models.objects.all()) < 4:
                        models = spiral_list_Models(spiral_list='', spiral_model_name=nameOfModel)
                        models.save()
                        form.save()
                        message = 'Uploaded'
                        error = ''

                        parent = 'media/files'

                        path_upload = os.path.join(parent, nameOfModel + '.py')
                        path_main = 'main/spiralupload.py'

                        update_upl_model(path_main, path_upload)


                    else:

                        delmodel=list(spiral_list_Models.objects.values_list('spiral_model_name', flat=True))[-1]
                        spiral_list_Models.objects.filter(spiral_model_name=delmodel).delete()

                        models = spiral_list_Models(spiral_list='', spiral_model_name=nameOfModel)
                        models.save()
                        form.save()

                        parent='media/files'
                        path = os.path.join(parent, delmodel+'.py')
                        # print(path)

                        os.remove(path)

                        message = 'Uploaded another model'
                        error = ''

                        path_upload = os.path.join(parent, nameOfModel + '.py')
                        path_main = 'main/spiralupload.py'

                        update_upl_model(path_main, path_upload)

                else:
                    message = ''
                    error = 'Uploaded another model'

                return render(request, "main/uploadModel.html",{"message":message,"error":error})

        except Exception:
            print(Exception)
            to_default(request)
            error = 'Follow the rules please'
            message = 'All models deleted'
            return render(request, "main/uploadModel.html", {"message": message, "error": error})

    else:
        form = DocumentForm()
    return render(request, 'main/uploadModel.html', {
        'form': form
    })

def testMethod(request):
    return render(request, 'main/testMethod.html')

def spiralModels(request):
    object_list = spiral_list_Models.objects.all()

    f = open('main/model.txt', 'w')

    f.write('')
    f.close()

    delete_from_folder('media/images/', 'jpg')
    delete_from_folder('media/images/', 'png')

    return render(request, 'main/spiralModels.html',{'object_list':object_list})
def microModels(request):
    object_list = micro_list_Models.objects.all()

    f = open('main/model.txt', 'w')
    f.write('')
    f.close()

    delete_from_folder('media/images/', 'jpg')
    delete_from_folder('media/images/', 'png')

    return render(request, 'main/microModels.html',{'object_list':object_list})

def spiralTest(request):
    request_p = request.build_absolute_uri()
    modelChoose = request_p[55:]
    print(modelChoose)
    f = open('main/model.txt','a')

    f.write(modelChoose)
    f.close()

    # check of request
    if request.method == "POST":
        form = ImageForm(data=request.POST, files=request.FILES)
        if form.is_valid():
            form.save()
            obj = form.instance

            DefaultModelList=['XGBClassifier','SupportVectorClassifier','RandomForestClassifier']

            f = open('main/model.txt', 'r')

            moddel=f.read()
            moddel.replace(' ', '')
            f.close()
            # check ny name which code need to be used
            # by default models of uploaded model
            if moddel not in DefaultModelList:
                pred,accuracy=spiral_model('main/datasetLogRF/spiral','media/images')

            else:
                pred,accuracy = picture_prediction(moddel)

            # getting answer form
            prediction=pred
            if prediction == 0:
                message="The patient does not have a predisposition to Parkinson's disease"
            else:message="The patient has a predisposition to Parkinson's disease"

            context={
                "obj": obj,
                "accuracy_score": 'Accuracy score is : '+str(accuracy),
                "prediction":message
            }

            return render(request, "main/spiralTest.html", context)
    else:
        form = ImageForm()
    img = Image.objects.all()

    return render(request, "main/spiralTest.html", {"modelChoose": modelChoose,"img": img, "form": form})


def test(request):

    request_p = request.build_absolute_uri()
    modelChoose = request_p[49:]


    f = open('main/model.txt', 'w')

    f.write('')
    f.close()

    error=''
    prediction_allert=''
    accuracy_score=''
    if request.method == 'POST':
        form = PatientForm(request.POST)

        print(form.is_valid())

        if form.is_valid():
           print(form.cleaned_data["name"])

           f = open('main/model.txt', 'a')

           f.write(modelChoose)
           f.close()

           f = open('main/model.txt', 'r')

           modelChoose = f.read()
           modelChoose.replace(' ', '')
           f.close()
           print(modelChoose)

           MicroDefault = ["DecisionTreeClassifier", "RandomForestClassifier", "SupportVectorClassifier"]

           #  check on which code has to be used
           # by default model or uploaded
           if modelChoose not in MicroDefault:
               prediction = micro_model\
                   ('main/parkinsons.data',
                     form.cleaned_data["MDVPFhi"], form.cleaned_data["MDVPFlo"],
                                        form.cleaned_data["MDVPJitterProcent"],
                                        form.cleaned_data["MDVPJitterABS"],
                                        form.cleaned_data["MDVPRAP"], form.cleaned_data["MDVPPPQ"],
                                        form.cleaned_data["JitterDDP"], form.cleaned_data["MDVPShimmer"],
                                        form.cleaned_data["MDVPShimmerdB"], form.cleaned_data["ShimmerAPQ3"],
                                        form.cleaned_data["ShimmerAPQ5"], form.cleaned_data["MDVDFo"],
                                        form.cleaned_data["MDVPAPQ"], form.cleaned_data["ShimmerDDA"],
                                        form.cleaned_data["NHR"], form.cleaned_data["HNR"],
                                        form.cleaned_data["RPDE"],
                                        form.cleaned_data["DFA"], form.cleaned_data["spread1"],
                                        form.cleaned_data["spread2"], form.cleaned_data["D2"],
                                        form.cleaned_data["PPE"])

           else:
            prediction = get_neural_prediction(modelChoose,form.cleaned_data["MDVPFhi"], form.cleaned_data["MDVPFlo"],
                                                   form.cleaned_data["MDVPJitterProcent"],
                                                   form.cleaned_data["MDVPJitterABS"],
                                                   form.cleaned_data["MDVPRAP"], form.cleaned_data["MDVPPPQ"],
                                                   form.cleaned_data["JitterDDP"], form.cleaned_data["MDVPShimmer"],
                                                   form.cleaned_data["MDVPShimmerdB"], form.cleaned_data["ShimmerAPQ3"],
                                                   form.cleaned_data["ShimmerAPQ5"], form.cleaned_data["MDVDFo"],
                                                   form.cleaned_data["MDVPAPQ"], form.cleaned_data["ShimmerDDA"],
                                                   form.cleaned_data["NHR"], form.cleaned_data["HNR"],
                                                   form.cleaned_data["RPDE"],
                                                   form.cleaned_data["DFA"],form.cleaned_data["spread1"],
                                                   form.cleaned_data["spread2"], form.cleaned_data["D2"],
                                                   form.cleaned_data["PPE"])


           if prediction[0] == 1:
               prediction_allert="The patient has a predisposition to Parkinson's disease"

           else:
               prediction_allert ="The patient does not have a predisposition to Parkinson's disease"

           accuracy_score = 'Accuracy score i s : ' + str(prediction[1])

           form.save()

        else:
             prediction_allert=''
             accuracy_score=''
             error='Not Right Form'

    # getting answer form
    form = PatientForm()
    context = {
        'form' : form,
        'error': error,
        'prediction_allert' : prediction_allert,
        'accuracy_score' : accuracy_score
    }
    return render(request, 'main/test.html',context)

# 192.168.31.196