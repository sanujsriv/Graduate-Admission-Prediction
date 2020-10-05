import pygame as pg
import joblib
import pandas as pd
import GRE_test



def main():

    boxes = [(715,536,148,56), (480,121,194,98)]
    take_test = [(709,61,151,58)]
#    print(boxes[1][0])
   
    screen = pg.display.set_mode((1100, 620))
    font = pg.font.Font(None, 32)
    clock = pg.time.Clock()
    blue = (0, 0, 0)
    img = pg.image.load("GRE_test/1.png")
    img2 = pg.image.load("GRE_test/2.jpg")
    pg.display.set_caption('Welcome to the graduate admission prediction system ')
    myfont = pg.font.Font(None, 30)    
    no  = myfont.render(" ",False,blue)
        
    input_box = pg.Rect(321, 75, 140, 32)
    input_box1 = pg.Rect(321, 140, 140, 32)
    input_box2 = pg.Rect(321, 200, 140, 32)
    input_box3 = pg.Rect(321, 266, 140, 32)
    input_box4 = pg.Rect(321, 322, 140, 32)
    color_inactive = pg.Color('lightblue')
    color_active = pg.Color('red')
    
    color = color_inactive
    color1 = color_inactive
    color2 = color_inactive
    color3 = color_inactive
    color4 = color_inactive
    
    active = False
    active1 = False
    active2 = False
    active3 = False
    active4 = False
    
    CGPA = ''
    GRE = ''
    TOEFL = ''
    University = ''
    Resarch = ''
    SOP = 5
    LOR =5
    done = False

    text = -1
    text1 = -1
    text2 = -1
    text3 = -1
    text4 = -1
    total =0

    menu = True
    test = True    
    

    while not done:

        while menu :
            scr = img2
            screen.blit(img2,(0,0))
            clock.tick(30)
            pg.display.update()

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    done = True
                    menu = False
            if event.type == pg.MOUSEBUTTONDOWN:
                mouse_pos = pg.mouse.get_pos()
                if mouse_pos[0]> boxes[1][0] and mouse_pos[0] < boxes[1][0] + boxes[1][2] and mouse_pos[1]> boxes[1][1] and mouse_pos[1] < boxes[1][1] + boxes[1][3]:
                     menu = False
                
            
        
        while test :
            scr = img
            screen.blit(img,(0,0))
            clock.tick(30)
            
            for event in pg.event.get():
                scr = img
                if event.type == pg.QUIT:
                    done = True
                    test = False
                if event.type == pg.MOUSEBUTTONDOWN:
                    mouse_pos = pg.mouse.get_pos()
                    for box in take_test :
                         if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:
                             total = main3()
                             
                        
                    for box in boxes :
                        if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:
                             if text == -1: # GRE
                                 if total != 0:
                                     text = total
                                 else :
                                     no = myfont.render("GRE field is empty!",False,blue)
                                     break
                             if text1 == -1: # TOEFL
                                 no = myfont.render("TOEFL field is empty!",False,blue)
                                 break
                             if text2 == -1: # University
                                 no = myfont.render("TOEFL field is empty!",False,blue)
                                 break

                             if text3 == -1: #CGPA
                                 no = myfont.render("CGPA field is empty!",False,blue)
                                 break

                             if text4 == -1: # Resarch
                                 no = myfont.render("Resarch field is empty!",False,blue)
                                 break
                             
                                                              
                             try : # GRE                          
                                 ft=int(text)
                                 print("GRE",ft)
                                 if(ft>340 or ft<260):
                                     no = myfont.render("Please enter a valid GRE score ranging between (260-340)",False,blue)
                                     break
                             except :
                                no = myfont.render("you entered string in GRE field, please just enter numbers!",False,blue)
                                break
                            
                             try :                                
                                 ft=int(text1)
                                 print("TOEFL",ft)
                                 if(ft>120 or ft<0):
                                
                                     no = myfont.render("Please enter a valid TOEFL score ranging between (0-120)",False,blue)
                                     break
                             except :
                                no = myfont.render("you entered string in TOEFL field, please just enter numbers!",False,blue)
                                break
                            
                             try :              
        
                                ft=int(text2)
                                print("UNI",ft)
                                if(ft>5 or ft<0):
                                 
                                     no = myfont.render("Enter the rating on scale of (0-5)",False,blue)
                                     break                              
                             except :
                                no = myfont.render("you entered string in University field , please just enter numbers!",False,blue)
                                break
                                                          
                             try :                               
                                ft= float(text3)
                                print("CGPA",ft)
                                if(ft>10.0 or ft<0.0):                                  
                                     no = myfont.render("CGPA value should be ranged between (0.0 to 10.0)",False,blue)
                                     break
                             except :
                                no = myfont.render("you entered string in CGPA, please just enter numbers!",False,blue)
                                break                            
                                

                             try :                          
                                 ft=int(text4)
                                 print("research",ft)
                                 if(ft>1 or ft<0):
                                  
                                     no = myfont.render("Please Enter 0 for No Exp & 1 if you have Research Exp",False,blue)
                                     break
                             except :
                                no = myfont.render("you entered string in Resarch field, please just enter numbers!",False,blue)
                                break

                            
                             my_model_loaded = joblib.load("models/RandomForest.pkl")
                             
                                               
                             test = [["1",(text),(text1),(text2),SOP,LOR,(text3),(text4)]]
                             df = pd.DataFrame(test, columns = [ "SN", 'GRE', 'TOEFL', 'u_rating', 'SOP', 'LOR', 'CGPA', 'Research'])
                             dec_tree_prediction=my_model_loaded.predict(df)
                             print("prediction",dec_tree_prediction)
                             

                             if dec_tree_prediction == 0:
                                  no = myfont.render("Sorry! You will not get the admission",False,blue)
                             elif dec_tree_prediction == 1:
                                 no = myfont.render("Congratulation! You will get the admission",False,blue)
                                 
                             #print("prediction", dec_tree_prediction)
                             
                             text = -1
                             text1 = -1
                             text2 = -1
                             text3 = -1
                             text4 = -1
                             total =0                          
                       
                                             
                            
                    # If the user clicked on the input_box rect.
                    if input_box.collidepoint(event.pos): #GRE
                        no = myfont.render(" ",False,blue)
                        # Toggle the active variable.
                        active = not active
                        color = color_active if active else color_inactive
                        active1 = False
                        active2 = False
                        active3 = False
                        active4 = False
                    elif input_box1.collidepoint(event.pos): # TOEFL
                        # Toggle the active variable.
                        active1 = not active1
                        color1 = color_active if active1 else color_inactive
                        active = False
                        active2 = False
                        active3 = False
                        active4 = False
                    elif input_box2.collidepoint(event.pos): # University
                        # Toggle the active variable.
                        active2 = not active2
                        color2 = color_active if active2 else color_inactive
                        active = False
                        active1 = False
                        active3 = False
                        active4 = False
                    elif input_box3.collidepoint(event.pos): # CGPA
                        # Toggle the active variable.
                        active3 = not active3
                        color3 = color_active if active3 else color_inactive
                        active1 = False
                        active2 = False
                        active = False
                        active4 = False
                    elif input_box4.collidepoint(event.pos): # Research
                        # Toggle the active variable.
                        active4 = not active4
                        color4 = color_active if active4 else color_inactive
                        active1 = False
                        active2 = False
                        active3 = False
                        active = False
                    
                    # Change the current color of the input box.
                    color = color_active if active else color_inactive
                    color1 = color_active if active1 else color_inactive
                    color2 = color_active if active2 else color_inactive
                    color3 = color_active if active3 else color_inactive
                    color4 = color_active if active4 else color_inactive
                if event.type == pg.KEYDOWN:
                            if active: # GRE
                                if event.key == pg.K_RETURN:
                                    #print(GRE)
                                    text = GRE
                                    GRE = ''                                    
                                elif event.key == pg.K_BACKSPACE:
                                    GRE = GRE[:-1]
                                else:
                                    GRE += event.unicode
                                    
                            elif active1: # TOEFL
                                if event.key == pg.K_RETURN:
                                    #print(TOEFL)
                                    text1 = TOEFL
                                    TOEFL = ''
                                elif event.key == pg.K_BACKSPACE:
                                    TOEFL = TOEFL[:-1]
                                else:
                                    TOEFL += event.unicode

                            elif active2:
                                if event.key == pg.K_RETURN:
                                    #print(University)
                                    text2 = University
                                    University = ''
                                elif event.key == pg.K_BACKSPACE:
                                    University = University[:-1]
                                else:
                                    University += event.unicode

                            elif active3:
                                if event.key == pg.K_RETURN:
                                    #print(CGPA)
                                    text3 = CGPA
                                    CGPA = ''
                                elif event.key == pg.K_BACKSPACE:
                                    CGPA = CGPA[:-1]
                                else:
                                    CGPA += event.unicode

                            elif active4:
                                if event.key == pg.K_RETURN:
                                    #print(Resarch)
                                    text4 = Resarch
                                    Resarch = ''
                                elif event.key == pg.K_BACKSPACE:
                                    Resarch = Resarch[:-1]
                                else:
                                    Resarch += event.unicode
                
                    
                

           
            # Render the current text.
            txt_surface = font.render(GRE, True, color)
            txt_surface1 = font.render(TOEFL, True, color1)
            txt_surface2 = font.render(University, True, color2)
            txt_surface3 = font.render(CGPA, True, color3)
            txt_surface4 = font.render(Resarch, True, color4)
            
            # Resize the box if the text is too long.
            width = max(200, txt_surface.get_width()+10)
            input_box.w = width
            input_box1.w = width
            input_box2.w = width
            input_box3.w = width
            input_box4.w = width
            # Blit the text.
            
            screen.blit(txt_surface, (input_box.x+5, input_box.y+5))
            screen.blit(txt_surface1, (input_box1.x+5, input_box1.y+5))
            screen.blit(txt_surface2, (input_box2.x+5, input_box2.y+5))
            screen.blit(txt_surface3, (input_box3.x+5, input_box3.y+5))
            screen.blit(txt_surface4, (input_box4.x+5, input_box4.y+5))
            screen.blit(no, (70,570))
            
            
            # Blit the input_box rect.
            pg.draw.rect(screen, color, input_box, 2)
            pg.draw.rect(screen, color1 , input_box1, 2)
            pg.draw.rect(screen, color2 , input_box2, 2)
            pg.draw.rect(screen, color3 , input_box3, 2)
            pg.draw.rect(screen, color4 , input_box4, 2)
            pg.display.update()
            
            
        screen.blit(scr,(0,0))
        pg.display.flip()
        clock.tick(30)
        
#
#def StringException(a):
#        if type(a)=='str':
#            return myfont.render("you entered string in CGPA field, please just enter numbers!",False,blue)


def main3():
    
    boxes = [(715,536,148,56), (717,552,170,90),(328,568,174,81)]

    ## reading 1
    boxes_reading = [(10,234,35,33), (10,295,38,35), (10,354,40,36),(11,416,36,31),(12,471,39,35)]
    Done_box = [328,568,174,81]
    pos_reading = [[26,252],[26,315] , [26,372],[25,432],[26,491]]

    ## reading 2
    boxes_reading1 = [(18,199,44,34), (19,287,48,39), (21,356,47,38)]

    ## SE1
    boxes_reading2 = [(27,249,26,22), (24,370,24,24),(20,430,33,21),(25,309,26,22), (20,492,29,24),(22,555,32,23)]    
    Done_box2 = [934,534,168,76]

    ## SE2
    boxes_reading3 = [(113,141,45,29), (111,203,54,27), (109,267,55,29),(113,333,48,27),(109,392,52,27),(107,458,56,24)]
    Done_box3 = [925,512,144,76]

    ## SE3
    boxes_reading4 = [(95,130,40,30), (94,203,41,24), (96,263,42,31),(96,333,45,34),(105,407,37,28),(108,467,40,28)]
    Done_box4 = [909,527,182,87]

    ## text1
    boxes_reading5 = [(52,313,26,22), (51,373,26,22), (51,432,24,24),(50,516,33,21),(49,577,29,24)]
    Done_box5 = [837,531,163,70]    

    ## text2
    boxes_reading6 = [(34,174,27,18), (31,210,30,18), (29,241,32,23)]
    boxes_reading7 = [(279,173,33,20),(278,209,33,19),(280,243,35,18)]
    Done_box6 = [835,513,180,85]
    
    ## text3
    boxes_reading8 = [(22,491,35,18), (20,524,35,16), (22,555,38,19)]
    boxes_reading9 = [(284,491,39,23),(287,527,36,16),(283,559,35,20)]

    ## text 4
    boxes_reading10 = [(27,188,28,20), (23,222,32,17), (22,257,32,16)]
    boxes_reading11 = [(163,189,33,20), (164,221,31,19), (163,252,34,19)]
    boxes_reading12 = [(308,187,41,23), (308,223,41,21), (309,256,39,20)]

    # quants 1
    boxes_reading13 = [(49,220,31,24), (49,259,34,26), (49,295,32,23),(46,332,35,27),(46,371,35,28)]
    boxes_reading14 = [(51,496,32,28), (50,534,32,28), (49,577,32,28)]

    # quants 2
    boxes_reading15 = [(54,89,24,15), (53,123,23,20), (53,163,22,16)]
    boxes_reading16 = [(61,353,20,16), (58,391,21,16), (58,426,22,19)]  
    
    
   
    screen = pg.display.set_mode((1125, 662))
    pg.init()
    #font = pg.font.Font("comic sans Ms", 32)
    clock = pg.time.Clock()
    blue = (0, 0, 0)
    img = pg.image.load("GRE_test/GRE_instructions.jpg")
    img2 = pg.image.load("GRE_test/GRE_reading_1.jpg")
    img3 = pg.image.load("GRE_test/GRE_reading_2.jpg")
    img4 = pg.image.load("GRE_test/SE1.PNG")
    img5 = pg.image.load("GRE_test/SE2.PNG")
    img6 = pg.image.load("GRE_test/SE3.PNG")
    img7 = pg.image.load("GRE_test/Text1.PNG")
    img8 = pg.image.load("GRE_test/Text2.PNG")
    img9 = pg.image.load("GRE_test/Text3.PNG")
    img10 = pg.image.load("GRE_test/Quants1.PNG")
    img11 = pg.image.load("GRE_test/Quants2.PNG")
    img12 = pg.image.load("GRE_test/Exit.jpg")
    pg.display.set_caption('Welcome to the graduate admission prediction system ')

    myfont = pg.font.Font(None, 30)
    myfont1 = pg.font.Font(None, 80)
    start_ticks=pg.time.get_ticks() #starter tick
    #no  = myfont.render(" ",False,blue)
        
    BLACK =(0,0,0)
    
    menu = True
    Exit = True
    done = False
    test = True
    test1 = True
    test2 = True
    test3 = True
    test4 = True
    test5 = True
    test6 = True
    test7 =True
    test8 = True
    test9 = True
    frames =0
    frames_second =0
    starting_timer = 20
    box2 =[0,0]
    box3 =[0,0]
    box4= [0,0]
    box5 =[0,0]
    box_ch = 0

    found_answer = []
    
    
    while not done:
         score = 0
         pg.display.set_caption('Welcome to the GRE test')
         while menu :             
            scr = img
            screen.blit(img,(0,0))
            clock.tick(30)
            pg.display.update()

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    done = True
                    Exit = False
                    test = False
                    test1 = False
                    test2 = False
                    test3 = False
                    test4 = False
                    test5 = False
                    test6 = False
                    test7 =False
                    test8 = False
                    test9= False
                    menu = False
                if event.type == pg.MOUSEBUTTONDOWN:
                    mouse_pos = pg.mouse.get_pos()
                    if mouse_pos[0]> boxes[1][0] and mouse_pos[0] < boxes[1][0] + boxes[1][2] and mouse_pos[1]> boxes[1][1] and mouse_pos[1] < boxes[1][1] + boxes[1][3]:
                       menu = False

            
         
         
         input_box = pg.Rect(0,0,0,0)
         while test :
            frames += 1
            frames_second +=1
            second = frames_second /30
            seconds= (frames /30)/60
            countdown = starting_timer - seconds
            #print("hi")
            scr = img2
            screen.blit(img2,(0,0))        
            
            for event in pg.event.get():
                scr = img2
                if event.type == pg.QUIT:
                    done = True
                    Exit = False
                    test = False
                    test1 = False
                    test2 = False
                    test3 = False
                    test4 = False
                    test5 = False
                    test6 = False
                    test7 =False
                    test8 = False
                    test9= False
                    menu = False
                if event.type == pg.MOUSEBUTTONDOWN:
                    mouse_pos = pg.mouse.get_pos()
                    
                    for box in boxes_reading :                         
                        if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:                            
                            box2 = box
                            input_box = pg.Rect(box[0], box[1],box[2],box[3])
                       
                    if mouse_pos[0]> Done_box[0] and mouse_pos[0] < Done_box[0] + Done_box[2] and mouse_pos[1]> Done_box[1] and mouse_pos[1] < Done_box[1] + Done_box[3]:
                        test = False
                        if box2[1] == 354:
                            score +=1
                        print("reading_score",score)
        
                        
            pg.draw.rect(screen, BLACK, input_box)
            if (60-second) < 0:
                frames_second = 0
            if (int(countdown ) == 0):
                done = False
                test = False
                test1 = False
                test2 = False
                test3 = False
                test4 = False
                test5 = False
                test6 = False
                test7 =False
                test8 = False
                test9= False                
            time_countdown = str(int(countdown )) +":" + str(int(60-second))
            time_label = myfont.render(time_countdown,True,(0,0,0))
            screen.blit(time_label,(1037,50))
            clock.tick(30)
            pg.display.update()
            
         
         input_box = pg.Rect(0,0,0,0)
         while test1 :
            frames += 1
            frames_second +=1
            second = frames_second /30            
            seconds= (frames /30)/60
            countdown = starting_timer - seconds
            #print("hi")
            scr = img3
            screen.blit(img3,(0,0))        
            
            for event in pg.event.get():
                scr = img3
                if event.type == pg.QUIT:
                    done = True
                    Exit = False
                    test = False
                    test1 = False
                    test2 = False
                    test3 = False
                    test4 = False
                    test5 = False
                    test6 = False
                    test7 =False
                    test8 = False
                    test9= False
                if event.type == pg.MOUSEBUTTONDOWN:
                    mouse_pos = pg.mouse.get_pos()                    
                    for box in boxes_reading1 :                         
                        if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:                            
                            input_box = pg.Rect(box[0], box[1],box[2],box[3])
                            pos_values1 = box
                            box2 = box
                        
                    if mouse_pos[0]> Done_box[0] and mouse_pos[0] < Done_box[0] + Done_box[2] and mouse_pos[1]> Done_box[1] and mouse_pos[1] < Done_box[1] + Done_box[3]:
                        test1 = False
                        if box2[1] == 199:
                            score +=1
                        print("reading2_score", score)
                                              
            pg.draw.rect(screen, BLACK, input_box )
            if (60-second) < 0:
                frames_second = 0
            if (int(countdown ) == 0):
                done = False
                test = False
                test1 = False
                test2 = False
                test3 = False
                test4 = False
                test5 = False
                test6 = False
                test7 =False
                test8 = False
                test9= False
            time_countdown = str(int(countdown )) +":" + str(int(60-second))
            time_label = myfont.render(time_countdown,True,(0,0,0))
            screen.blit(time_label,(1037,50))            
            clock.tick(30)
            pg.display.update()
        
         BLACK = (0,0,0)
         input_box = pg.Rect(0,0,0,0)
         input_box1= pg.Rect(0,0,0,0)
         done_box1 = True
         while test2 :
            frames += 1
            frames_second +=1
            second = frames_second /30
            seconds= (frames /30)/60
            countdown = starting_timer - seconds  
            #print("hi")
            scr = img4
            screen.blit(img4,(0,0))        
            
            for event in pg.event.get():
                scr = img4
                if event.type == pg.QUIT:
                    done = True
                    Exit = False
                    test = False
                    test1 = False
                    test2 = False
                    test3 = False
                    test4 = False
                    test5 = False
                    test6 = False
                    test7 =False
                    test8 = False
                    test9= False
                
                if event.type == pg.MOUSEBUTTONDOWN:
                    mouse_pos = pg.mouse.get_pos()
                    if done_box1 == True:
                        for i, box in enumerate(boxes_reading2) :                         
                            if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:
                                
                                input_box = pg.Rect(box[0], box[1],box[2],box[3])
                                pos_values1 = box                                                         
                                box2 = box                                
                                del( boxes_reading2[i])                               
                                done_box1 = False
                    
                         
                    for box in boxes_reading2:                         
                        if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:                            
                                                     
                            input_box1 = pg.Rect(box[0], box[1],box[2],box[3])
                            pos_values1 = box                            
                            box3 = box
                            boxes_reading2.append(box2)
                            done_box1 = True
                            

                #print("found_answer",box2)
                #print(box2)
                                                
                if mouse_pos[0]> Done_box2[0] and mouse_pos[0] < Done_box2[0] + Done_box2[2] and mouse_pos[1]> Done_box2[1] and mouse_pos[1] < Done_box2[1] + Done_box2[3]:
                    test2 = False
                    done_box1 = True                    
                    if box2[0] == 20 or box2[0] == 24 :
                        score +=1                        
                    if box3[0] == 24 or box3[0] == 20:
                        score +=1
                    print("score",score)
                    
                
                            
                        
            BLACK = (0,0,0)                                  
            pg.draw.rect(screen, BLACK, input_box )
            pg.draw.rect(screen, BLACK, input_box1 )
            
            if (60-second) < 0:
                frames_second = 0
            if (int(countdown ) == 0):
                done = False
                test = False
                test1 = False
                test2 = False
                test3 = False
                test4 = False
                test5 = False
                test6 = False
                test7 =False
                test8 = False
                test9= False
            time_countdown = str(int(countdown )) +":" + str(int(60-second))
            time_label = myfont.render(time_countdown,True,(0,0,0))
            screen.blit(time_label,(1037,50))            
            clock.tick(30)
            pg.display.update()


         
         input_box = pg.Rect(0,0,0,0)
         input_box1  = pg.Rect(0,0,0,0)
         while test3 :
            frames += 1
            frames_second +=1
            second = frames_second /30
            seconds= (frames /30)/60
            countdown = starting_timer - seconds  
            #print("hi")
            scr = img5
            screen.blit(img5,(0,0))        
            
            for event in pg.event.get():
                scr = img5
                if event.type == pg.QUIT:
                    done = True
                    Exit = False
                    test = False
                    test1 = False
                    test2 = False
                    test3 = False
                    test4 = False
                    test5 = False
                    test6 = False
                    test7 =False
                    test8 = False
                    test9= False
                if event.type == pg.MOUSEBUTTONDOWN:
                    mouse_pos = pg.mouse.get_pos()
                    if done_box1 == True:
                        for i, box in enumerate(boxes_reading3) :                         
                            if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:                                
                                input_box = pg.Rect(box[0], box[1],box[2],box[3])
                                pos_values1 = box
                                box2 = box                                
                                del( boxes_reading3[i])                               
                                done_box1 = False
                                
                    for box in boxes_reading3:                         
                        if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:                                                   
                            input_box1 = pg.Rect(box[0], box[1],box[2],box[3])
                            pos_values1 = box                            
                            box3 = box
                            boxes_reading3.append(box2)
                            done_box1 = True
                        
                    if mouse_pos[0]> Done_box3[0] and mouse_pos[0] < Done_box3[0] + Done_box3[2] and mouse_pos[1]> Done_box3[1] and mouse_pos[1] < Done_box3[1] + Done_box3[3]:
                        test3 = False
                        done_box1 = True                        
                        if box2[0] == 111 or box2[0] == 113 :
                            score +=1                        
                        if box3[0] == 113 or box3[0] == 111:
                            score +=1
                        print("score",score)
                        
                                             
            pg.draw.rect(screen, BLACK, input_box )
            pg.draw.rect(screen, BLACK, input_box1 )
            if (60-second) < 0:
                frames_second =0
            if (int(countdown ) == 0):
                done = False
                test = False
                test1 = False
                test2 = False
                test3 = False
                test4 = False
                test5 = False
                test6 = False
                test7 =False
                test8 = False
                test9= False
            time_countdown = str(int(countdown )) +":" + str(int(60-second))
            time_label = myfont.render(time_countdown,True,(0,0,0))
            screen.blit(time_label,(1037,50))    
            clock.tick(30)
            pg.display.update()

         
         input_box = pg.Rect(0,0,0,0)
         input_box1 = pg.Rect(0,0,0,0)
         done_box1 =True
         while test4 :
            frames += 1
            frames_second +=1
            second = frames_second /30
            seconds= (frames /30)/60
            countdown = starting_timer - seconds  
            #print("hi")
            scr = img6
            screen.blit(img6,(0,0))        
            
            for event in pg.event.get():
                scr = img6
                if event.type == pg.QUIT:
                    done = True
                    Exit = False
                    test = False
                    test1 = False
                    test2 = False
                    test3 = False
                    test4 = False
                    test5 = False
                    test6 = False
                    test7 =False
                    test8 = False
                    test9= False
                if event.type == pg.MOUSEBUTTONDOWN:
                    mouse_pos = pg.mouse.get_pos()
                    if done_box1 == True:                  
                        for i,box in enumerate(boxes_reading4 ):                         
                            if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:                                
                                input_box = pg.Rect(box[0], box[1],box[2],box[3])
                                pos_values1 = box
                                box2 = box                                
                                del( boxes_reading4[i])                               
                                done_box1 = False                                
                                
                    for box in boxes_reading4:
                         if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:                                                   
                            input_box1 = pg.Rect(box[0], box[1],box[2],box[3])
                            pos_values1 = box                            
                            box3 = box
                            boxes_reading4.append(box2)
                            done_box1 = True
                            
                            
                    if mouse_pos[0]> Done_box4[0] and mouse_pos[0] < Done_box4[0] + Done_box4[2] and mouse_pos[1]> Done_box4[1] and mouse_pos[1] < Done_box4[1] + Done_box4[3]:
                        test4 = False
                        done_box1 = True
                        if box2[0] == 96 or box2[0] == 108 :
                            score +=1                        
                        if box3[0] == 96 or box3[0] == 108:
                            score +=1
                        print("score",score)
                        
                                              
            pg.draw.rect(screen, BLACK, input_box )
            pg.draw.rect(screen, BLACK, input_box1 )
            if (60-second) < 0:
                frames_second = 0
            if (int(countdown ) == 0):
                done = False
                test = False
                test1 = False
                test2 = False
                test3 = False
                test4 = False
                test5 = False
                test6 = False
                test7 =False
                test8 = False
                test9= False
            time_countdown = str(int(countdown )) +":" + str(int(60-second))
            time_label = myfont.render(time_countdown,True,(0,0,0))
            screen.blit(time_label,(1037,50))    
            clock.tick(30)
            pg.display.update()



         BLACK = (255,255,255)
         input_box = pg.Rect(0,0,0,0)
         while test5 :
            frames += 1
            frames_second +=1
            second = frames_second /30
            seconds= (frames /30)/60
            countdown = starting_timer - seconds
            #print("hi")
            scr = img7
            screen.blit(img7,(0,0))        
            
            for event in pg.event.get():
                scr = img7
                if event.type == pg.QUIT:
                    done = True
                    Exit = False
                    test = False
                    test1 = False
                    test2 = False
                    test3 = False
                    test4 = False
                    test5 = False
                    test6 = False
                    test7 =False
                    test8 = False
                    test9= False
                if event.type == pg.MOUSEBUTTONDOWN:
                    mouse_pos = pg.mouse.get_pos()                    
                    for box in boxes_reading5 :                         
                        if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:
                            BLACK =(0,0,0)
                            input_box = pg.Rect(box[0], box[1],box[2],box[3])
                            pos_values1 = box
                           # print(pos_values )
                            box2 =box
                        
                    if mouse_pos[0]> Done_box5[0] and mouse_pos[0] < Done_box5[0] + Done_box5[2] and mouse_pos[1]> Done_box5[1] and mouse_pos[1] < Done_box5[1] + Done_box5[3]:
                        test5 = False
                        if box2[0] == 51:
                            score +=1
                        print("score",score)
                        
                        
                                              
            pg.draw.rect(screen, BLACK, input_box )
            if (60-second) < 0:
                frames_second = 0
            if (int(countdown ) == 0):
                done = False
                test = False
                test1 = False
                test2 = False
                test3 = False
                test4 = False
                test5 = False
                test6 = False
                test7 =False
                test8 = False
                test9= False
            time_countdown = str(int(countdown )) +":" + str(int(60-second))
            time_label = myfont.render(time_countdown,True,(0,0,0))
            screen.blit(time_label,(1037,50))    
            clock.tick(30)
            pg.display.update()


         
         input_box = pg.Rect(0,0,0,0)
         input_box1 = pg.Rect(0,0,0,0)
         input_box2 = pg.Rect(0,0,0,0)
         input_box3 = pg.Rect(0,0,0,0)
         
         while test6 :
            frames += 1
            frames_second +=1
            second = frames_second /30
            seconds= (frames /30)/60
            countdown = starting_timer - seconds
            #print("hi")
            scr = img8
            screen.blit(img8,(0,0))         
            for event in pg.event.get():
                scr = img8
                if event.type == pg.QUIT:
                    done = True
                    Exit = False
                    test = False
                    test1 = False
                    test2 = False
                    test3 = False
                    test4 = False
                    test5 = False
                    test6 = False
                    test7 =False
                    test8 = False
                    test9= False
                if event.type == pg.MOUSEBUTTONDOWN:
                    mouse_pos = pg.mouse.get_pos()                    
                    for box in boxes_reading6 :                         
                        if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:                             
                            input_box = pg.Rect(box[0], box[1],box[2],box[3])
                            box2 = box
                            pos_values1 = box                            
                    for box in boxes_reading7 :                         
                        if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:                            
                            input_box1 = pg.Rect(box[0], box[1],box[2],box[3])
                            pos_values1 = box
                            box3 = box
                    for box in boxes_reading8 :                         
                        if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:                            
                            input_box2 = pg.Rect(box[0], box[1],box[2],box[3])
                            pos_values1 = box
                            box4 = box
                    for box in boxes_reading9 :                         
                        if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:                           
                            input_box3 = pg.Rect(box[0], box[1],box[2],box[3])
                            pos_values1 = box
                            box5 = box
                    
                        
                    if mouse_pos[0]> Done_box6[0] and mouse_pos[0] < Done_box6[0] + Done_box6[2] and mouse_pos[1]> Done_box6[1] and mouse_pos[1] < Done_box6[1] + Done_box6[3]:
                        test6 = False
                        
                        if box2[1] == 174:
                            score +=1
                        if box3[1] == 209:
                            score +=1
                        if box4[1] == 555:
                            score +=1
                        if box5[1] == 527:
                            score +=1
                        print("Q2_tx_score",score)
                        
                        
            BLACK =(0,0,0)                                  
            pg.draw.rect(screen, BLACK, input_box )
            pg.draw.rect(screen, BLACK, input_box1 )
            pg.draw.rect(screen, BLACK, input_box2 )
            pg.draw.rect(screen, BLACK, input_box3 )
            if (60-second) < 0:
                frames_second = 0
            if (int(countdown ) == 0):
                done = False
                test = False
                test1 = False
                test2 = False
                test3 = False
                test4 = False
                test5 = False
                test6 = False
                test7 =False
                test8 = False
                test9= False
            time_countdown = str(int(countdown )) +":" + str(int(60-second))
            time_label = myfont.render(time_countdown,True,(0,0,0))
            screen.blit(time_label,(1037,50))   
            clock.tick(30)
            pg.display.update()
 
         BLACK = (255,255,255)
         input_box = pg.Rect(0,0,0,0)
         input_box1 = pg.Rect(0,0,0,0)
         input_box2= pg.Rect(0,0,0,0)
         
         
         while test7 :
            frames += 1
            frames_second +=1
            second = frames_second /30
            seconds= (frames /30)/60
            countdown = starting_timer - seconds            
            #print("hi")
            scr = img9
            screen.blit(img9,(0,0))         
            for event in pg.event.get():
                scr = img9
                if event.type == pg.QUIT:
                    done = True
                    Exit = False
                    test = False
                    test1 = False
                    test2 = False
                    test3 = False
                    test4 = False
                    test5 = False
                    test6 = False
                    test7 =False
                    test8 = False
                    test9= False
                if event.type == pg.MOUSEBUTTONDOWN:
                    mouse_pos = pg.mouse.get_pos()                    
                    for box in boxes_reading10 :                         
                        if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:                            
                            input_box = pg.Rect(box[0], box[1],box[2],box[3])
                            pos_values1 = box                            
                            box2 =box

                    for box in boxes_reading11 :                         
                        if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:                           
                            input_box1 = pg.Rect(box[0], box[1],box[2],box[3])
                            pos_values1 = box
                            box3 =box
                            
                            
                    for box in boxes_reading12 :                         
                        if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:                            
                            input_box2 = pg.Rect(box[0], box[1],box[2],box[3])
                            pos_values1 = box
                            box4 =box
                                    
                        
                    if mouse_pos[0]> Done_box6[0] and mouse_pos[0] < Done_box6[0] + Done_box6[2] and mouse_pos[1]> Done_box6[1] and mouse_pos[1] < Done_box6[1] + Done_box6[3]:
                        test7 = False
                        if box2[1] == 188:
                            score +=1
                        if box3[1] == 189:
                            score +=1
                        if box4[1] == 223:
                            score +=1
                        
                        print("Q2_tx_score",score)
                        
            BLACK =(0,0,0)                                  
            pg.draw.rect(screen, BLACK, input_box )
            pg.draw.rect(screen, BLACK, input_box1 )
            pg.draw.rect(screen, BLACK, input_box2 )
            if (60-second) < 0:
                frames_second = 0
            if (int(countdown ) == 0):
                done = False
                test = False
                test1 = False
                test2 = False
                test3 = False
                test4 = False
                test5 = False
                test6 = False
                test7 =False
                test8 = False
                test9= False
            time_countdown = str(int(countdown )) +":" + str(int(60-second))
            time_label = myfont.render(time_countdown,True,(0,0,0))
            screen.blit(time_label,(1037,50))   
            clock.tick(30)
            pg.display.update()

         BLACK = (255,255,255)
         input_box = pg.Rect(0,0,0,0)
         input_box1 = pg.Rect(0,0,0,0)
         while test8:
            frames += 1
            frames_second +=1
            second = frames_second /30
            seconds= (frames /30)/60
            countdown = starting_timer - seconds            
            scr = img10
            screen.blit(img10,(0,0))         
            for event in pg.event.get():
                scr = img10
                if event.type == pg.QUIT:
                    done = True
                    Exit = False
                    test = False
                    test1 = False
                    test2 = False
                    test3 = False
                    test4 = False
                    test5 = False
                    test6 = False
                    test7 =False
                    test8 = False
                    test9= False
                if event.type == pg.MOUSEBUTTONDOWN:
                    mouse_pos = pg.mouse.get_pos()                    
                    for box in boxes_reading13 :                         
                        if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:
                            input_box = pg.Rect(box[0], box[1],box[2],box[3])
                            pos_values1 = box
                            box2 = box
                    for box in boxes_reading14 :                         
                        if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:
                            input_box1 = pg.Rect(box[0], box[1],box[2],box[3])
                            pos_values1 = box
                            box3 = box   
                        
                    if mouse_pos[0]> Done_box6[0] and mouse_pos[0] < Done_box6[0] + Done_box6[2] and mouse_pos[1]> Done_box6[1] and mouse_pos[1] < Done_box6[1] + Done_box6[3]:
                        test8 = False
                        if box2[1] == 259:
                            score +=1
                        if box3[1] == 496:
                            score +=1                         
                        print("Q2_quant",score)
                        
            BLACK = (0,0,0)                                 
            pg.draw.rect(screen, BLACK, input_box )
            pg.draw.rect(screen, BLACK, input_box1 )
            if (60-second) < 0:
                frames_second = 0
            if (int(countdown ) == 0):
                done = False
                test = False
                test1 = False
                test2 = False
                test3 = False
                test4 = False
                test5 = False
                test6 = False
                test7 =False
                test8 = False
                test9= False
            time_countdown = str(int(countdown )) +":" + str(int(60-second))
            time_label = myfont.render(time_countdown,True,(0,0,0))
            screen.blit(time_label,(1037,50))  
            clock.tick(30)
            pg.display.update()

         BLACK = (255,255,255)
         input_box = pg.Rect(0,0,0,0)
         input_box1 = pg.Rect(0,0,0,0)
         while test9:
            frames += 1
            frames_second +=1
            second = frames_second /30
            seconds= (frames /30)/60
            countdown = starting_timer - seconds            
            scr = img10
            screen.blit(img11,(0,0))         
            for event in pg.event.get():
                scr = img10
                if event.type == pg.QUIT:
                    done = True
                    Exit = False
                    test = False
                    test1 = False
                    test2 = False
                    test3 = False
                    test4 = False
                    test5 = False
                    test6 = False
                    test7 =False
                    test8 = False
                    test9 = False
                if event.type == pg.MOUSEBUTTONDOWN:
                    mouse_pos = pg.mouse.get_pos()                    
                    for box in boxes_reading15 :                         
                        if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:
                            input_box = pg.Rect(box[0], box[1],box[2],box[3])
                            pos_values1 = box
                            box2 = box
                    for box in boxes_reading16 :                         
                        if mouse_pos[0]> box[0] and mouse_pos[0] < box[0] + box[2] and mouse_pos[1]> box[1] and mouse_pos[1] < box[1] + box[3]:
                            input_box1 = pg.Rect(box[0], box[1],box[2],box[3])
                            pos_values1 = box
                            box3 = box   
                        
                    if mouse_pos[0]> Done_box6[0] and mouse_pos[0] < Done_box6[0] + Done_box6[2] and mouse_pos[1]> Done_box6[1] and mouse_pos[1] < Done_box6[1] + Done_box6[3]:
                        test9 = False
                        if box2[1] == 89:
                            score +=1
                        if box3[1] == 391:# b
                            score +=1                         
                        print("Q3_4_quant",score)
                        
            BLACK = (0,0,0)                                 
            pg.draw.rect(screen, BLACK, input_box )
            pg.draw.rect(screen, BLACK, input_box1 )
            if (60-second) < 0:
                frames_second = 0
            if (int(countdown ) == 0):
                done = False
                test = False
                test1 = False
                test2 = False
                test3 = False
                test4 = False
                test5 = False
                test6 = False
                test7 =False
                test8 = False
                test9 = False
            time_countdown = str(int(countdown )) +":" + str(int(60-second))
            time_label = myfont.render(time_countdown,True,(0,0,0))
            screen.blit(time_label,(1037,50))  
            clock.tick(30)
            pg.display.update()

         while Exit :             
            scr = img11
            screen.blit(img12,(0,0))
            score_label = myfont1.render(str(int((4*score)+260)),True,(0,0,0))
            screen.blit(score_label,(625,287))  
            clock.tick(30)
            pg.display.update()

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    done = True
                    Exit = False
                    test = False
                    test1 = False
                    test2 = False
                    test3 = False
                    test4 = False
                    test5 = False
                    test6 = False
                    test7 =False
                    test8 = False
                    test9 =False
                if event.type == pg.MOUSEBUTTONDOWN:
                    mouse_pos = pg.mouse.get_pos()
                    if mouse_pos[0]> boxes[1][0] and mouse_pos[0] < boxes[1][0] + boxes[1][2] and mouse_pos[1]> boxes[1][1] and mouse_pos[1] < boxes[1][1] + boxes[1][3]:
                       Exit = False
                       done = True
         return int((4*score)+260)

if __name__ == '__main__':
    pg.init()
    main()
    pg.quit()
