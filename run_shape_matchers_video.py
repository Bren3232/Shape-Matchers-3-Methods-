
# Copyright (C) 2024 Brendan Murphy - All Rights Reserved
# This file is part of the "Shape Matchers (3 Methods)" project.
# See the LICENSE.TXT file which should be included in the same directory as this file.



import cvtools_short as cvt
import cv2
import numpy as np
import imutils as im
import os
# import time

cap = cv2.VideoCapture(0)

save_tempImg = "template_chart_img.png"          # is img of chart
save_bin_tempImg = "template_bin_img.png"   # is binary img
save_tempTxt = "template.txt"
save_tempProp = "template_prop.txt"

# If 0 keyboard "q" exits, if not, current camera image is used as the template.
# When setting template have 1 contour in center of frame.
set_template = 1

use_average_prop = 0    # Better accuracy, save a chart template before using

# Methods to use
do_chart_match = 0
do_prop_list_match = 0
do_rot_match = 0


onc = 0

# Make the frame binary

# Low HSV limits
l_h = 0
l_s = 0
l_v = 0

# Upper HSV limits
u_h = 255
u_s = 255
u_v = 70      # For black objects on white background


if not os.path.isfile(save_tempTxt):
    with open(save_tempTxt, 'a'):
        pass
if not os.path.isfile(save_tempProp):
    with open(save_tempProp, 'a'):
        pass


while True:

    _, frame = cap.read()
    # frame = cv2.imread(save_bin_tempImg)    # gets obscured in this loop

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # hsv = frame.copy()


    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_b, u_b)

    res = cv2.bitwise_and(frame, frame, mask=mask)

#     bilateral = cv2.bilateralFilter(res, 15, 75, 75)   #@ added
#     kernel = np.ones((5, 5), np.float32) / 225
#     smoothed = cv2.filter2D(res, -1, kernel)             #@ added

#     cv2.imshow('bilateral Blur', bilateral)             #@ added
#     cv2.imshow('Averaging', smoothed)                   #@ added
    # cv2.imshow("frame", frame)
    # cv2.imshow("mask", mask)
    cv2.imshow("Cam view hit 'q' to set as template", res)

    # Make binary
    # res = cv2.dilate(res, (7, 7), iterations=2)  # Optional for smoothing
    # res = cv2.erode(res, (5, 5), iterations=2)

    bin = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    bin[bin[:, :] != 0] = 255

    # cv2.imshow("Binary", bin)

    # time.sleep(1)
    # cv2.waitKey(0)


    if cv2.waitKey(1) & 0xFF == ord('q'):

        if set_template != 0:
            # for loop on img list here

            bin = im.rotate(bin, None, None, 1.2)

            # t5 = time.perf_counter()
            # front_s = len(cn[0]) // 2
            front_s = 40  # for some reason really has to be 40
            mid_s = front_s // 2
            step = 4  # stepping at is good but makes chart longer and so chart match charts are longer and take more time
            # step2 = 4

            # Make a prop list for each rotation of the template, then average it and/or add the charts together
            if use_average_prop == 1:
                r_step = 0
                r_step_amount = 15
                prop_dict = {}
                ctt = 0

                # Get prop lists for many rotations and store in a dict
                for idx, i in enumerate(range(round(360 / r_step_amount))):
                    # bin_r = cv2.imread(save_bin_tempImg, 0)   # for testing
                    bin_r = bin.copy()
                    bin_r = im.rotate(bin_r, r_step)

                    cv2.imshow("bin  r", bin_r)
                    cv2.waitKey(0)

                    cont = cvt.find_cont(bin_r)
                    img2, prop_list = cvt.contour_chain_prop(bin_r, cont[0], retur="height", front_start=front_s,
                                                             middle_start=mid_s, step=step, view=False)

                    # Keep shortest length to downsize to
                    if r_step == 0:
                        shorty = len(prop_list)
                    else:
                        shorty = min(shorty, len(prop_list))

                    prop_dict[str(r_step)] = prop_list

                    ctt += 1
                    r_step += r_step_amount


                # Cut them all down to the same size, shorty is an int
                prop_lists = []
                for idx, i in enumerate(prop_dict.items()):
                    # if idx == 0:

                    if len(i[1]) != shorty:
                        ap = cvt.fan_delete_resize(i[1], shorty)
                        prop_lists.append(ap)
                    else:
                        prop_lists.append(i[1])

                    # i = cvt.fan_delete_resize(i, shorty)

                # def simple_chart(prop_lists, bin, color, avg=None):          # A visual check unimportant
                #
                #     ch = np.zeros_like(bin)
                #     for i in prop_lists:
                #         # print(i[:10])
                #         for idx, j in enumerate(i):
                #             cv2.circle(ch, (idx * 2 + 100, int(j) + 100), 1, color, -1)
                #         cv2.imshow("ch test", ch)
                #         cv2.waitKey(0)
                #         # ch = np.zeros_like(bin)
                #     # if avg != None:
                #     #     for idx, i in enumerate(avg):
                #     #         cv2.circle(ch, (idx * 2 + 100, int(i) + 100), 1, 255, -1)
                #     #         cv2.imshow("chhh test", ch)
                #     #         cv2.waitKey(0)
                #     return ch

                # simple_chart(prop_lists, bin, 255)

                prop_lists_aligned = []
                tem = prop_lists[0]

                prop_lists = np.asarray(prop_lists)

                # Roll them to best match using prop_list_match_best_roll
                for idx, i in enumerate(prop_lists):

                    if idx == 0:                        # might just save a bit of time
                        prop_lists_aligned.append(i)
                        continue

                    # For best accuracy temp_posi_h from the txt file should be an average also
                    # fixme this reads from save_tempTxt when it might be emtpy
                    best_roll_amt = cvt.prop_list_match_best_roll(prop_list=i, temp_prop_list=tem,
                                                                    template_text=save_tempTxt, roll=1)

                    tt = np.roll(i, best_roll_amt)  # different result xx This one is working
                    prop_lists_aligned.append(tt)


                # chh = simple_chart(prop_lists_aligned, bin, 120)   # A visual check unimportant

                prop_lists_aligned = np.asarray(prop_lists_aligned)
                # print(prop_lists_aligned.shape)


                # --- Now the prop lists are all sized, and best match aligned, time to do averaging and overlaying ---
                prop_list_avg = np.sum(prop_lists_aligned, axis=0) / prop_lists_aligned.shape[0]
                # print(prop_list_avg.shape)
                # cv2.waitKey(0)

                # A visual check unimportant
                # # Don't use prop_lists down here it's messed, simple_chart() is overlaying
                # for idx, i in enumerate(prop_list_avg):
                #     cv2.circle(chh, (idx * 2 + 100, int(i) + 100), 1, 255, -1)
                #     cv2.imshow("chhh test", chh)
                #     cv2.waitKey(0)

                # ** Now prop_list_avg is ready for prop_list_match(), but chart_match_set_template first

                prop_list = prop_list_avg      #


                # would be saved to file
                # temp_prop_list = prop_list
                f = open(save_tempProp, "w+")
                for i in prop_list:
                    f.write(str(i) + "\n")
                f.close()

                cv2.imwrite(save_bin_tempImg, bin)

                # img2, prop_list = ccc.contour_chain_circle(img, cont[0], retur="circle", radius=mid_s, step=step2, view=False)

                # print("Time for contour_chain_prop ", time.perf_counter() - t5)
                cv2.imshow("img used as template", bin)
                cv2.imshow("scaled template img", bin)
                # cv2.waitKey(0)

                # # # USING cvt2 also
                # cvt.chart_match_set_template(prop_list=prop_list, contours=cont, save_image_as=save_tempImg,
                #                              save_text_as=save_tempTxt, save_front_start=front_s)

                # Takes prop_lists_aligned not the single prop_list, overlays all
                cvt.chart_match_set_template_overlay(prop_lists_aligned=prop_lists_aligned, contours=cont,
                                    save_image_as=save_tempImg, save_text_as=save_tempTxt, save_front_start=front_s)


            else:
                cont = cvt.find_cont(bin)  # CHAIN APPROX SIMPLE with step=5 even has good results
                # img[:, :] = np.where(img[:, :] > 127, 255, 0)

                img2, prop_list = cvt.contour_chain_prop(bin, cont[0], retur="height", front_start=front_s,
                                                         middle_start=mid_s, step=step, view=False)

                # sys.exit()

                # would be saved to file
                # temp_prop_list = prop_list
                f = open(save_tempProp, "w+")
                for i in prop_list:
                    f.write(str(i) + "\n")
                f.close()

                cv2.imwrite(save_bin_tempImg, bin)

                # img2, prop_list = ccc.contour_chain_circle(img, cont[0], retur="circle", radius=mid_s, step=step2, view=False)

                # print("Time for contour_chain_prop ", time.perf_counter() - t5)
                cv2.imshow("img used as template", bin)
                cv2.imshow("scaled template img", bin)
                # cv2.waitKey(0)

                # # USING cvt2 also
                cvt.chart_match_set_template(prop_list=prop_list, contours=cont, save_image_as=save_tempImg,
                                             save_text_as=save_tempTxt, save_front_start=front_s)

        break


    if do_chart_match != 0 or do_prop_list_match != 0:
        contours = cvt.find_cont(bin)    # changed contourArea size to 200
        # front_s = 40
        # mid_s = 20

        step = 4  # lower steps are better on F imgs, with temp steps at 4   grow 4 is good on F
        croll = 4  # May 14 everything seems good croll 6 on F takes 7ms per contour

        show = cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR)
        # show[show == 255] = 70    # darken to see txt better
        show2 = show.copy()

        # cv2.drawContours(img, contours, -1, (0, 25, 255), 3)

        img56 = np.zeros_like(bin)  # don't really need
        cv2.drawContours(img56, contours, -1, 255, -1)  # xx  Can't fill conts individually, doesn't take cont[0]
        # cv2.fillPoly(img56, c, 255)                            # so no sense putting in loop
        # img56[:, :] = np.where(img56[:, :] > 127, 255, 0)
        # cv2.imshow("img before loop", img56)
        # removed chart_match, now calling chart_match chart_match

        count = 0

        # ---------- Chart Match 2 -----------
        # if count == 0:
        if onc == 0:
            # Get other template data from txt file
            td = open(save_tempTxt, "r")
            # td = open("t-many.txt", "r")
            t_data = td.read().splitlines()
            td.close()

            t_cont_len = float(t_data[3])
            t_cont_area = float(t_data[4])
            t_cont_radius = float(t_data[5])  # radius of minEnclosingCircle of template
            t_front_start = int(t_data[6])
            # print('t_front_start from txt file ', t_front_start)

            tp = open(save_tempProp, "r")
            temp_prop = tp.read().splitlines()
            tp.close()
            temp_prop_list = [float(x) for x in temp_prop]
            # print("temp prop list type: ", type(temp_prop_list))

            # Just for visual
            temp_img = cv2.imread(save_bin_tempImg, 0)
            temp_img2 = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2BGR)
            cv2.putText(temp_img2, "Template Image", (180, temp_img.shape[0] - 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 180), 2)
            cv2.imshow("Template Image", temp_img2)
            onc = 1

        # Splitting chart_match into another for loop for time comparison ** chart_match is 0.16s when chart match 1 is 0.16
        # t222 = time.time()
        # t555 = time.perf_counter()
        for c in contours[:]:
            if cv2.contourArea(c) < 200:
                continue
            count += 1

            # xx Plus 10 here gets a gap of 15 ** good improvements especially for chart_match, still more experimenting to do
            # try basing on minEnclosing circle and/or cont area ***  chart_match with grow 1 needs this to work
            # ..now plus nothing is better

            cont_area = cv2.contourArea(c)
            (x, y), rad = cv2.minEnclosingCircle(c)

            ## Adding t_front_start * 0.1 at end is improvement
            ## front_s = int((len(c) / t_cont_len * t_front_start) + (t_front_start * 0.1))    #+ 3  # good
            ## front_s = int(len(c) / t_cont_len * t_front_start)
            ## front_s = int(cont_area / t_cont_area * t_front_start)    # no for F, ok for apr1-centered template
            front_s = int((rad / t_cont_radius * t_front_start) + (
                        t_front_start * 0.1))  # also good maybe to average of 2, maybe add ~3?
            ## front_s = int(rad / t_cont_radius * t_front_start)

            # front_s = 40

            ## front_s = cont_area // area_ratio + 0
            ## front_s = rad // int(radius_ratio) + 0
            ## front_s = len(c) // 29 + 0  # this works here but not in  chart match1, and reduces gap from 9 to 5
            mid_s = front_s // 2  # 29 is len of F template cont // 40,  ** plus some num like 5 and great improve for both **
            ## print(front_s, "  ", mid_s)       # Now +0
            ## print("front_s in for chart_match  ", front_s)

            try:
                img2, prop_list = cvt.contour_chain_prop(img56, c, retur="height", front_start=front_s, middle_start=mid_s,
                                                         step=step, view=False)
            except:
                print(" \n    Faulted in contour_chain_prop   \n  ")
                continue

            # img2, prop_list = ccc.contour_chain_circle(img56, c, retur="circle", radius=mid_s, step=step2, view=False)

            if do_chart_match != 0:
                over2 = cvt.chart_match(prop_list=prop_list, template_image=save_tempImg, template_text=save_tempTxt,
                                     roll=croll, warp_match_x=True, warp_match_y=True, zones=0, view=False)
            else:
                # Now returns be cos similarity   span of 19 on f mix   beats chart_match speed by 0.01s and span by a lot
                # Is the best and fastest way
                try:
                    over2 = cvt.prop_list_match(prop_list=prop_list, temp_prop_list=temp_prop_list, template_text=save_tempTxt,
                                                roll=croll)
                except:
                    print("  \n  Faulted in prop_list_match (likely sample cont too big) \n   ")
                    continue

            # over2 = cm3.chart_match3(prop_list=prop_list, template_image=save_tempImg, temp_prop_list=temp_prop_list,
            #                          template_text=save_tempTxt, roll=croll, warp_match_x=True, warp_match_y=True,
            #                          zones=0, view=False)
            # span 5 usually on f mixed, with grows
            # poor accuracy on alphabet samples

            # print("over from chart match2 ", over2) #, "  Time ", time.time() - t22)

            if over2 > 70:
                col2 = (0, 210, 0)
            else:
                col2 = (0, 0, 180)

            # if over2 > 90:
            #     col2 = (0, 210, 0)
            # elif over2 > 80 and over2 < 90:
            #     col2 = (0, 150, 120)
            # elif over2 > 70 and over2 < 80:
            #     col2 = (0, 110, 140)
            # else:
            #     col2 = (0, 0, 180)
            # col2 = (0, int(over2 * 2.5), 255 - (int(over2 * 2.5)))

            # print(f'Order: {count}')
            # print("Best cos: ", round(over2, 3))  # span of 19 on f mix

            # cv2.putText(show2, f'({count})', c[0][0], cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, col2)
            # cv2.putText(show2, str(round(over2, 3)), (c[0][0][0], c[0][0][1] + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, col2, 1)
            cv2.putText(show2, str(round(over2, 2)), c[0][0], cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, col2, 1)

            # cv2.putText(show2, str(count) + ": " + str(round(over2, 3)), c[0][0], cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, col2)

            # cv2.imshow("imgg --> show d", show)
            cv2.imshow("imgg --> show2  d <----", show2)
            # cv2.waitKey(0)

    # time.sleep(1)

    if do_rot_match == 1:
        # show3 = show.copy()
        show3 = cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR)

        if onc == 0 or onc == 1:
            temp_img_rot = cv2.imread(save_bin_tempImg, 0)
            temp_img2 = cv2.cvtColor(temp_img_rot, cv2.COLOR_GRAY2BGR)
            cv2.putText(temp_img2, "Template Image", (180, temp_img_rot.shape[0] - 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 180), 2)
            cv2.imshow("Template Image", temp_img2)
            temp_c_rot = cvt.find_cont(temp_img_rot)

            # Center the template, draw filled, and get scale value
            temp_cent_img, _, temp_scaleV = cvt.center_contour(img=temp_img_rot, cont=temp_c_rot, contour_thickness=-1,
                                                           center_based_on=0, scale_value=1)
            onc = 2

        contours = cvt.find_cont(bin)

        for c in contours:
            if cv2.contourArea(c) < 200:
                continue

            img99 = np.zeros_like(bin)
            cv2.drawContours(img99, c, -1, 255, -1)
            co = cvt.find_cont(img99)  # just for better visual, adds 0.01s to time
            # cv2.imshow("img99", img99)
            # cv2.waitKey(0)

            # Center the sample, draw filled, and get scale value
            imgc, cc, scaleV = cvt.center_contour(img=img99, cont=co, contour_thickness=-1, center_based_on=0,
                                                  scale_value=1)

            # Scale sample image to template image size
            scale_ratio = temp_scaleV / scaleV
            samp_img = im.rotate(imgc, None, None, scale_ratio)

            # cv2.imshow("samp img", samp_img)
            # cv2.waitKey(0)

            # Rotate the sample image, over the template image, while subtracting to get a best accuracy reading
            best_acc, best_acc_deg = cvt.match_best_rotation(template_img=temp_cent_img, sample_img=samp_img, step=5,
                                                             break_accuracy=101000, subtract=False, view=False)

            # TODO try adding ellipse to center_based_on, and scale_value in cvt.center_contour

            if best_acc > 60:
                col2 = (0, 210, 0)
            else:
                col2 = (0, 0, 180)

            # # For average of 2
            # try:
            #     best_acc = round((over2 + best_acc) / 2, 2)
            #
            #
            #     cv2.putText(show3, str(best_acc), c[0][0], cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, col2)
            #     cv2.putText(show3, f" Rotated {str(best_acc_deg)} degrees", (120, temp_img_rot.shape[0] - 50),
            #                 cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 180), 2)
            # except:
            #     pass

            # Just rot match
            cv2.putText(show3, str(best_acc), c[0][0], cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, col2)
            cv2.putText(show3, f" Rotated {str(best_acc_deg)} degrees", (120, temp_img_rot.shape[0] - 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 180), 2)

            # print("best accuracy from match best rotation ", best_acc)

            # cv2.imshow("centered contour", imgc)
            # cv2.waitKey(0)

        # cv2.imshow("template img", img)
        # cv2.imshow("imgg --> show", show)
        # cv2.imshow("imgg --> show2 <----", show2)
        cv2.imshow("imgg --> show3 match best rotation<---", show3)
        # cv2.waitKey(0)

# cap.release()
cv2.destroyAllWindows()


