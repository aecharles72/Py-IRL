#!/usr/bin/env python

import sys
import random
import itertools
import numpy as np
import cv2 as cv
#pref order Py Standard Lib, 3rd party, user-defined


MAP_FILE = 'cape_python.png'

SA1_CORNERS = (130, 265, 180, 315) # (Upper Left X, Upper Left Y, Lower Right X, Lower Right Y)
SA2_CORNERS = (80, 255, 130, 305)
SA3_CORNERS = (105, 205, 155, 255)

#search and rescue mission classes

class Search():
    """Bayesian Search & Rescue game with 3 search areas."""

    # define initial attribute values for class object
    def __init__(self,name):
        self.name = name
        self.img = cv.imread(MAP_FILE, cv.IMREAD_COLOR)
        if self.img is None:
            print('Could not load map file {}'.format(MAP_FILE), file=sys.stderr)
            sys.exit(1)

        self.area_actual = 0
        self.sunksub_actual = [0, 0] # As "local" coords within search area

        self.sa1 = self.img[SA1_CORNERS[1] : SA1_CORNERS[3],
                            SA1_CORNERS[0] : SA1_CORNERS[2]]  # NumPy req the range be provided in UL Y : LR Y, UL X : LR X

        self.sa2 = self.img[SA2_CORNERS[1] : SA2_CORNERS[3],
                            SA2_CORNERS[0] : SA2_CORNERS[2]]

        self.sa3 = self.img[SA3_CORNERS[1] : SA3_CORNERS[3],
                            SA3_CORNERS[0] : SA3_CORNERS[2]]

        self.p1 = 0.2
        self.p2 = 0.5
        self.p3 = 0.3

        self.sep1 = 0
        self.sep2 = 0
        self.sep3 = 0

    #define draw map method taking self and sunksub's last known pos
    def draw_map(self, last_known):
        """Display basemap with scale, last known xy location, search areas."""
        cv.line(self.img, (20, 370), (70, 370), (0, 0, 0), 2) #draws scale bar
        cv.putText(self.img, '0', (8,370), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0)) #attribute, actual text, coord tuple, font name, font cale, color tuple
        cv.putText(self.img, '50 Nautical Miles', (71,370), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))

        cv.rectangle(self.img, (SA1_CORNERS[0], SA1_CORNERS[1]), (SA1_CORNERS[2], SA1_CORNERS[3]), (0, 0, 0), 1) # map image, four corners, color tuple, line weight
        cv.putText(self.img, '1', (SA1_CORNERS[0] + 3, SA1_CORNERS[1] + 15), cv.FONT_HERSHEY_PLAIN, 1, 0) # puts search area number inside upper left corner of box  UL X, UL Y
        cv.rectangle(self.img, (SA2_CORNERS[0], SA2_CORNERS[1]), (SA2_CORNERS[2], SA2_CORNERS[3]), (0, 0 , 0), 1)
        cv.putText(self.img, '2', (SA2_CORNERS[0] + 3, SA2_CORNERS[1] + 15), cv.FONT_HERSHEY_PLAIN, 1, 0)
        cv.rectangle(self.img, (SA3_CORNERS[0], SA3_CORNERS[1]), (SA3_CORNERS[2], SA3_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(self.img, '3', (SA3_CORNERS[0] + 3, SA3_CORNERS[1] + 15), cv.FONT_HERSHEY_PLAIN, 1, 0)

        cv.putText(self.img, '+', (last_known), cv.FONT_HERSHEY_PLAIN, 1, (0,0,255))  # post symbol for last known position, OpenCV used blue-green-red color format insead of rgb

        # legend for last known and actual positions
        cv.putText(self.img, '+ = Last Known Position', (274, 355), cv.FONT_HERSHEY_PLAIN, 1, (0,0,255))
        cv.putText(self.img, '* = Actual Position', (275, 370), cv.FONT_HERSHEY_PLAIN, 1, (255,0,0))

        cv.imshow('Search Area', self.img)
        cv.moveWindow('Search Area', 2000, 10) # displays in upper right corner of monitor, may need to adjust coords for diff monitor
        cv.waitKey(500)

    # define method to randomly choose sub's actual loc taking self and number of search areas being used
    def sunksub_final_location(self, num_search_areas):
        """Return the actual x,y location of the missing sunksub."""
        # find the sunksub coords with respect to any Search Area subarray, selecting [0] with random choice means rows are used  [1] means columns
        self.sunksub_actual[0] = np.random.choice(self.sa1.shape[1])  # shape attribute must be a tuple with as many elements as dimension in the array
        self.sunksub_actual[1] = np.random.choice(self.sa1.shape[0])  #  to get coords of array within shape >>> print(np.shape(self,SA1))

        area = int(random.triangular(1, num_search_areas + 1))  # args are low and high endpoints

        #convert local search area coords to global coords of full base map and update search area attribute
        if area == 1:
            x = self.sunksub_actual[0] + SA1_CORNERS[0]
            y = self.sunksub_actual[1] + SA1_CORNERS[1]
            self.area_actual = 1
        elif area == 2:
            x = self.sunksub_actual[0] + SA2_CORNERS[0]
            y = self.sunksub_actual[1] + SA2_CORNERS[1]
            self.area_actual = 2
        elif area == 3:
            x = self.sunksub_actual[0] + SA3_CORNERS[0]
            y = self.sunksub_actual[1] + SA3_CORNERS[1]
            self.area_actual = 3
        return x, y


    # define method to calculate th effectiveness of given search
    def calc_search_effectiveness(self):
        """Set decimal search effectiveness value per search area."""
        self.sep1 = random.uniform(0.2, 0.9)  # will always search at least 20% but never more than 90% of area
        self.sep2 = random.uniform(0.2, 0.9)
        self.sep3 = random.uniform(0.2, 0.9)

    #define method to conduct search taking object itself, area number chosen by user, subarray for chosen area, randomly chosen search effectiveness value
    def conduct_search(self, area_num, area_array, effectiveness_prob):
        """Return search results and list of searched coordinates."""
        local_y_range = range(area_array.shape[0])
        local_x_range = range(area_array.shape[1])
        coords = list(itertools.product(local_x_range, local_y_range)) #product returns typles of all permutations-with-repetition for given seq  #to see it work >>> import itertools  >>> x_range = [1,2,3]  >>> y_range = [4,5,6]  >>> coords = list(itertools.product(x_range, y_range))  >>> coords
        random.shuffle(coords) #shuffle so do not continuously search the same end of list
        coords = coords[:int((len(coords) * effectiveness_prob))] # trim list based on search effectiveness probability
        loc_actual = (self.sunksub_actual[0], self.sunksub_actual[1])

        # check if sunksub is found
        if area_num == self.area_actual and loc_actual in coords:
            return 'Found in Area {}'.format(area_num), coords
        else:
            return 'Not Found', coords


    # define method to update target probabilities
    def revise_target_probs(self):
        """Update area target probabilities based on search effectiveness."""  # break Bayes equation into two parts
        denom = self.p1 * (1 - self.sep1) + self.p2 * (1 - self.sep2) + self.p3 * (1 - self.sep3)
        self.p1 = self.p1 * (1 - self.sep1) / denom
        self.p2 = self.p2 * (1 - self.sep2) / denom
        self.p3 = self.p3 * (1 - self.sep3) / denom

# display GUI menu to run game
def draw_menu(search_num):
    """Print menu of choices for conducting area searches."""
    print('\nSearch {}'.format(search_num))
    print(                                  # using """ with print function displays the menu
        """

        Choose next areas to search:

        0 - Quit
        1 - Search Area 1 twice
        2 - Search Area 2 twice
        3 - Search Area 3 twice
        4 - Search Areas 1 & 2
        5 - Search Areas 1 & 3
        6 - Search Areas 2 & 3
        7 - Start over
        """
        )

# define main function to run program
def main():
    app = Search('Cape_Python')  # name object
    app.draw_map(last_known=(160, 290)) # draw map and pass last known position
    sunksub_x, sunksub_y = app.sunksub_final_location(num_search_areas=3) # get sunksub location and pass number of search areas
    print("-" * 65)
    print("\nInitial Target (P) Probabilities:")
    print("P1 = {:.3f}, P2 = {:.3f}, P3 = {:.3f}".format(app.p1, app.p2, app.p3)) # print initial or prior target probs
    search_num = 1 # keeps track of conducted searches

    #start loop to run game, will loop till user chooses to exit
    while True:
        app.calc_search_effectiveness()
        draw_menu(search_num)
        choice = input("Choice: ")

        if choice == "0":
            sys.exit()
        elif choice == "1":
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1)
            results_2, coords_2 = app.conduct_search(1, app.sa1, app.sep1)
            app.sep1 = (len(set(coords_1 + coords_2))) / (len(app.sa1)**2)
            app.sep2 = 0
            app.sep3 = 0
        elif choice == "2":
            results_1, coords_1 = app.conduct_search(2, app.sa2, app.sep2)
            results_2, coords_2 = app.conduct_search(2, app.sa2, app.sep2)
            app.sep1 = 0
            app.sep2 = (len(set(coords_1 + coords_2))) / (len(app.sa2)**2)
            app.sep3 = 0
        elif choice == "3":
            results_1, coords_1 = app.conduct_search(3, app.sa3, app.sep3)
            results_2, coords_2 = app.conduct_search(3, app.sa3, app.sep3)
            app.sep1 = 0
            app.sep2 = 0
            app.sep3 = (len(set(coords_1 + coords_2))) / (len(app.sa3)**2)
        elif choice == "4":
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1)
            results_2, coords_1 = app.conduct_search(2, app.sa2, app.sep2)
            app.sep3 = 0
        elif choice == "5":
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1)
            results_2, coords_2 = app.conduct_search(3, app.sa3, app.sep3)
            app.sep2 = 0
        elif choice == "6":
            results_1, coords_1 = app.conduct_search(2, app.sa2, app.sep2)
            results_2, coords_2 = app.conduct_search(3, app.sa3, app.sep3)
            app.sep1 = 0
        elif choice == "7": #reset game
            main()

        else:
            print("\nSorry this isn't a valid choice.", file=sys.stderr)
            continue

        app.revise_target_probs() # use Bayes rule to update target probs

        # display search results and sep
        print("\nSearch {} Results 1 = {}".format(search_num, results_1), file=sys.stderr)
        print("\Search {} Results 2 = {}\n".format(search_num, results_2), file=sys.stderr)
        print("Search {} Effectiveness (E):".format(search_num))
        print("E1 = {:.3f}, E2 = {:.3f}, E3 = {:.3f}".format(app.sep1, app.sep2, app.sep3))

        if results_1 == 'Not Found' and results_2 == 'Not Found' :
            print("\nNew Target Probabilites (P) for Search {}:".format(search_num + 1))
            print("P1 = {:.3f}, P2 = {:.3f}, P3 = {:.3f}".format(app.p1, app.p2, app.p3))
        else:
            cv.circle(app.img, (sunksub_x, sunksub_y), 3, (255, 0, 0), -1)  # draw a circle, pass base map image, set sunksub tuple for center point, set radius in pixels, color, line weight ( neg value fills circle with color)
            cv.imshow('Search Area', app.img)
            cv.waitKey(1500)
            main()
        search_num += 1

if __name__ == '__main__':
    main()
