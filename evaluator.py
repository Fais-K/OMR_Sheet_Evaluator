import numpy as np 
import argparse
from imutils import contours
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
ap.add_argument("-t", "--style", default="standard", help="style of the OMR sheet")
args = vars(ap.parse_args())

cv2.namedWindow("final", cv2.WINDOW_NORMAL)

alphabet = {i-1:chr(i+64) for i in range(1, 27)}

info_blocks = {

	# "barcode" : 0,
	"name" : 1,
	"set" : 2,
	"roll_number" : 3,
	"subject" : 4,
	"test_id" : 5,
	"mobile_no" : 6
}

quest_sets = {

	"quest_set1" : 7,
	"quest_set2" : 8,
	"quest_set3" : 9
}

answer_set1 = {

	0 : 0,
	1 : 2,
	2 : 1,
	3 : 2,
	4 : 0,
	5 : 1,
	6 : 1,
	7 : 2,
	8 : 0,
	9 : 2
}
answer_set2 = {

	0 : 1,
	1 : 0,
	2 : 2,
	3 : 1,
	4 : 2,
	5 : 2,
	6 : 1,
	7 : 3,
	8 : 0,
	9 : 0
}
answer_set3 = {

	0 : 2,
	1 : 1,
	2 : 0,
	3 : 0,
	4 : 3,
	5 : 0,
	6 : 1,
	7 : 1,
	8 : 1,
	9 : 2
}


def sort_contours(cnts, method="tb/lr"):
	if method == "tb/lr":
		bounding_rects = [cv2.boundingRect(c) for c in cnts]
		(cnts, bounding_rects) = zip(*sorted(zip(cnts, bounding_rects), key=lambda b: (b[1][1], b[1][0])))
	elif method == "lr/tb":
		bounding_rects = [cv2.boundingRect(c) for c in cnts]
		(cnts, bounding_rects) = zip(*sorted(zip(cnts, bounding_rects), key=lambda b: (b[1][0], b[1][1])))
	else:
		print("Unable to process the specified method.")

	return cnts, bounding_rects

def find_info(img, cnts, block):
	cnts, bounding_rects = sort_contours(cnts)
	if block == "quest_set1" or block == "quest_set2" or block == "quest_set3":
		x, y, w, h = bounding_rects[quest_sets[block]]
	else:
		x, y, w, h = bounding_rects[info_blocks[block]]

	image_roi = img[y:y+h, x:x+w]
	gray_roi = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)

	thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

	inner_cnts = cv2.findContours(thresh_roi.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	inner_cnts = imutils.grab_contours(inner_cnts)

	mask_one = np.zeros(image_roi.shape[:2], np.uint8)

	for c in inner_cnts:
		x, y, w, h = cv2.boundingRect(c)
		ar = w / float(h)

		if w >= 10 and w <= 12 and h >= 10 and h <= 12 and ar >= 0.9 and ar <= 1.1:
			cv2.drawContours(mask_one, [c], 0, (255,255,255), -1)

	bubble_cnts = []

	bubble_cnts = cv2.findContours(mask_one.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	bubble_cnts = imutils.grab_contours(bubble_cnts)

	if block == "name" or block == "mobile_no":
		bubble_cnts = bubble_cnts[:len(bubble_cnts)-1]

	bubble_cnts, bubble_bounding_rects = sort_contours(bubble_cnts, "lr/tb")

	bubble_bounding_rects = list(bubble_bounding_rects)

	mask_two = np.zeros(image_roi.shape[:2], np.uint8)

	x, y, w, h = bubble_bounding_rects[0]
	cX = x + w // 2
	cY = y + h // 2

	cv2.circle(mask_two, (cX, cY), 5, 255, -1)
	
	for i in range(len(bubble_bounding_rects)-1):
		prev_x, prev_y, prev_w, prev_h = bubble_bounding_rects[i]
		prev_cX = prev_x + prev_w // 2
		prev_cY = prev_y + prev_h // 2

		for j in range(i+1, i+2):
			x, y, w, h = bubble_bounding_rects[j]
			cX = x + w // 2
			cY = y + h // 2

			diff = abs(cX - prev_cX)

			if diff == 0:
				cv2.circle(mask_two, (cX, cY), 5, 255, -1)

			elif diff > 0 and diff <= 3:
				bubble_bounding_rects[j] = (prev_x, y, prev_w, h)
				cv2.circle(mask_two, (prev_cX, cY), 5, 255, -1)

			else:
				cv2.circle(mask_two, (cX, cY), 5, 255, -1)

	bubble_cnts = cv2.findContours(mask_two.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	bubble_cnts = imutils.grab_contours(bubble_cnts)

	bubble_cnts, bubble_bounding_rects = sort_contours(bubble_cnts, "lr/tb")

	if block == "name":
		bubble_set = np.arange(26, len(bubble_cnts)+26, 26)
		blank_pass = 0
		name = []

		for e, l in enumerate(bubble_set):
			blank_pass += 1
			if blank_pass > 2:
				break

			for i in range(l-26, l):
				mask = np.zeros(gray_roi.shape, dtype="uint8")
				cv2.drawContours(mask, [bubble_cnts[i]], -1, 255, -1)
				mask = cv2.bitwise_and(thresh_roi, thresh_roi, mask=mask)
				total = cv2.countNonZero(mask)
			
				if total > 75:
					blank_pass -= 1
					key = i - e * 26
					name.append(alphabet[key])
					break
		name = "".join(name)
		return name

	elif block == "set":
		set_code = []
		for i in range(len(bubble_cnts)):
			mask = np.zeros(gray_roi.shape, dtype="uint8")
			cv2.drawContours(mask, [bubble_cnts[i]], -1, 255, -1)
			mask = cv2.bitwise_and(thresh_roi, thresh_roi, mask=mask)
			total = cv2.countNonZero(mask)
		
			if total > 75:
				key = i 
				set_code.append(alphabet[key])
				break
		set_code = "".join(set_code)
		return set_code

	elif block == "roll_number" or block == "subject" or block == "test_id" or block == "mobile_no":
		bubble_set = np.arange(10, len(bubble_cnts)+10, 10)
		number = []
		for e, l in enumerate(bubble_set):
			for i in range(l-10, l):
				mask = np.zeros(gray_roi.shape, dtype="uint8")
				cv2.drawContours(mask, [bubble_cnts[i]], -1, 255, -1)
				mask = cv2.bitwise_and(thresh_roi, thresh_roi, mask=mask)
				total = cv2.countNonZero(mask)

				if total > 75:
					value = i - e * 10
					number.append(value)
					break
		res = "".join(map(str, number))
		return res

	else:
		bubble_cnts, bubble_bounding_rects = sort_contours(bubble_cnts)
		bubble_set = np.arange(4, len(bubble_cnts)+4, 4)
		not_attended = 0
		correct = 0
		wrong = 0
		for e, l in enumerate(bubble_set):
			blank_pass = 1
			for i in range(l-4, l):
				mask = np.zeros(gray_roi.shape, dtype="uint8")
				cv2.drawContours(mask, [bubble_cnts[i]], -1, 255, -1)
				mask = cv2.bitwise_and(thresh_roi, thresh_roi, mask=mask)
				total = cv2.countNonZero(mask)

				if total > 75:
					blank_pass = 0
					value = i - e * 4

					if block == "quest_set1":
						answers = answer_set1
					elif block == "quest_set2":
						answers = answer_set2
					else:
						answers = answer_set3

					if answers[e] == value:
						correct += 1
					else:
						wrong += 1
					break
			if blank_pass == 1:
				not_attended += 1
		return correct, wrong, not_attended


image = cv2.imread(args["image"])
style = args["style"]

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
edged = cv2.Canny(blur_image, 40, 60)
edged = cv2.dilate(edged, None, iterations=1)

block_cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
block_cnts = imutils.grab_contours(block_cnts)
block_cnts = sorted(block_cnts, key=cv2.contourArea, reverse=True)[:10]

infos = []

final = np.ones((image.shape[0]-90, image.shape[1], 3))

for block in info_blocks.keys():
	infos.append(find_info(image, block_cnts, block))

cv2.putText(final, "STUDENT DETAILS:", (25, 50), cv2.FONT_HERSHEY_PLAIN, 1.8, (50, 0, 100), 2)
cv2.putText(final, "NAME: {}".format(infos[0]), (30, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, (150, 0, 0), 1)
cv2.putText(final, "SET: {}".format(infos[1]), (30, 150), cv2.FONT_HERSHEY_PLAIN, 1.5, (150, 0, 0), 1)
cv2.putText(final, "ROLL NUMBER: {}".format(infos[2]), (30, 200), cv2.FONT_HERSHEY_PLAIN, 1.8, (150, 0, 0), 1)
cv2.putText(final, "SUBJECT: {}".format(infos[3]), (30, 250), cv2.FONT_HERSHEY_PLAIN, 1.5, (150, 0, 0), 1)
cv2.putText(final, "TEST ID: {}".format(infos[4]), (30, 300), cv2.FONT_HERSHEY_PLAIN, 1.5, (150, 0, 0), 1)
cv2.putText(final, "MOBILE NO: {}".format(infos[5]), (30, 350), cv2.FONT_HERSHEY_PLAIN, 1.5, (150, 0, 0), 1)

correct_ans = []
wrong_ans = []
not_attended = []

for quest in quest_sets:
	correct, wrong, not_attended_qs = find_info(image, block_cnts, quest)
	correct_ans.append(correct)
	wrong_ans.append(wrong)
	not_attended.append(not_attended_qs)

part1_score = correct_ans[0] * 1 - wrong_ans[0] * 0
part2_score = correct_ans[1] * 2 - wrong_ans[1] * 0
part3_score = correct_ans[2] * 4 - wrong_ans[2] * 1 

total_score = part1_score + part2_score + part3_score

cv2.putText(final, "MARKING SCHEME:", (25, 400), cv2.FONT_HERSHEY_PLAIN, 1.8, (50, 0, 100), 2)
cv2.putText(final, "[PART-1] CORRECT = 1, WRONG = 0, NOT ATTENDED = 0", (30, 450), cv2.FONT_HERSHEY_PLAIN, 1.5, (150, 0, 0), 1)
cv2.putText(final, "[PART-2] CORRECT = 2, WRONG = 0, NOT ATTENDED = 0", (30, 500), cv2.FONT_HERSHEY_PLAIN, 1.5, (150, 0, 0), 1)
cv2.putText(final, "[PART-3] CORRECT = 4, WRONG = -1, NOT ATTENDED = 0", (30, 550), cv2.FONT_HERSHEY_PLAIN, 1.5, (150, 0, 0), 1)


cv2.putText(final, "EXAM SCORES:", (25, 600), cv2.FONT_HERSHEY_PLAIN, 1.8, (50, 0, 100), 2)
cv2.putText(final, "PART-1 SCORE: {} [CORRECT={}, WRONG={}, NOT ATTENDED={}]".format(part1_score, correct_ans[0], wrong_ans[0], not_attended[0]), (30, 650), cv2.FONT_HERSHEY_PLAIN, 1.5, (150, 0, 0), 1)
cv2.putText(final, "PART-2 SCORE: {} [CORRECT={}, WRONG={}, NOT ATTENDED={}]".format(part2_score, correct_ans[1], wrong_ans[1], not_attended[1]), (30, 700), cv2.FONT_HERSHEY_PLAIN, 1.5, (150, 0, 0), 1)
cv2.putText(final, "PART-3 SCORE: {} [CORRECT={}, WRONG={}, NOT ATTENDED={}]".format(part3_score, correct_ans[2], wrong_ans[2], not_attended[2]), (30, 750), cv2.FONT_HERSHEY_PLAIN, 1.5, (150, 0, 0), 1)
cv2.putText(final, "TOTAL SCORE: {}".format(total_score), (30, 800), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 150), 1)

cv2.imshow("final", final)
cv2.waitKey(0)
