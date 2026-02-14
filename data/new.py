from yt_dlp import YoutubeDL

# ---------------- SONG LIST ----------------
songs = [

    ("Take Me Home, Country Roads", "John Denver", 0.96, 0.78),
("Annie's Song", "John Denver", 0.92, 0.70),
("Sunshine on My Shoulders", "John Denver", 0.94, 0.68),
("Rocky Mountain High", "John Denver", 0.96, 0.80),
("Back Home Again", "John Denver", 0.92, 0.74),

("Ring of Fire", "Johnny Cash", 0.92, 0.86),
("Folsom Prison Blues", "Johnny Cash", 0.88, 0.82),
("I Walk the Line", "Johnny Cash", 0.90, 0.76),
("A Boy Named Sue", "Johnny Cash", 0.94, 0.88),
("Man in Black", "Johnny Cash", 0.86, 0.78),

("Jolene", "Dolly Parton", 0.88, 0.82),
("9 to 5", "Dolly Parton", 0.96, 0.90),
("Coat of Many Colors", "Dolly Parton", 0.92, 0.70),
("Here You Come Again", "Dolly Parton", 0.94, 0.78),
("Islands in the Stream", "Dolly Parton", 0.96, 0.82),

("Friends in Low Places", "Garth Brooks", 0.98, 0.88),
("The Dance", "Garth Brooks", 0.90, 0.70),
("If Tomorrow Never Comes", "Garth Brooks", 0.92, 0.72),
("Thunder Rolls", "Garth Brooks", 0.86, 0.80),
("Ain't Goin' Down ('Til the Sun Comes Up)", "Garth Brooks", 0.96, 0.92),

("Before He Cheats", "Carrie Underwood", 0.94, 0.88),
("Jesus, Take the Wheel", "Carrie Underwood", 0.90, 0.74),
("Cowboy Casanova", "Carrie Underwood", 0.92, 0.86),
("Blown Away", "Carrie Underwood", 0.88, 0.90),
("Something in the Water", "Carrie Underwood", 0.94, 0.82),

("Live Like You Were Dying", "Tim McGraw", 0.92, 0.70),
("Humble and Kind", "Tim McGraw", 0.94, 0.68),
("My Best Friend", "Tim McGraw", 0.96, 0.76),
("Just to See You Smile", "Tim McGraw", 0.90, 0.72),
("Something Like That", "Tim McGraw", 0.98, 0.88),

("Blue Ain't Your Color", "Keith Urban", 0.88, 0.68),
("Somebody Like You", "Keith Urban", 0.96, 0.84),
("Days Go By", "Keith Urban", 0.98, 0.88),
("You'll Think of Me", "Keith Urban", 0.90, 0.72),
("Wasted Time", "Keith Urban", 0.94, 0.82),

("Need You Now", "Lady A", 0.92, 0.70),
("Just a Kiss", "Lady A", 0.94, 0.76),
("Downtown", "Lady A", 0.96, 0.86),
("American Honey", "Lady A", 0.90, 0.72),
("Bartender", "Lady A", 0.96, 0.88),

("Cruise", "Florida Georgia Line", 0.98, 0.92),
("H.O.L.Y.", "Florida Georgia Line", 0.92, 0.74),
("This Is How We Roll", "Florida Georgia Line", 0.96, 0.90),
("Simple", "Florida Georgia Line", 0.94, 0.82),
("Stay", "Florida Georgia Line", 0.88, 0.70),



("The Gambler", "Kenny Rogers", 0.92, 0.78),
("Lucille", "Kenny Rogers", 0.86, 0.72),
("Coward of the County", "Kenny Rogers", 0.88, 0.74),
("Lady", "Kenny Rogers", 0.90, 0.70),
("Through the Years", "Kenny Rogers", 0.92, 0.68),

("Amarillo by Morning", "George Strait", 0.94, 0.76),
("Check Yes or No", "George Strait", 0.96, 0.82),
("The Chair", "George Strait", 0.90, 0.70),
("Carrying Your Love with Me", "George Strait", 0.92, 0.74),
("Write This Down", "George Strait", 0.94, 0.78),

("Boot Scootin' Boogie", "Brooks & Dunn", 0.98, 0.92),
("Neon Moon", "Brooks & Dunn", 0.88, 0.70),
("My Maria", "Brooks & Dunn", 0.96, 0.82),
("Only in America", "Brooks & Dunn", 0.98, 0.88),
("Red Dirt Road", "Brooks & Dunn", 0.92, 0.72),

("Chicken Fried", "Zac Brown Band", 0.98, 0.86),
("Colder Weather", "Zac Brown Band", 0.90, 0.72),
("Free", "Zac Brown Band", 0.94, 0.74),
("Homegrown", "Zac Brown Band", 0.96, 0.84),
("Knee Deep", "Zac Brown Band", 0.98, 0.88),

("My Church", "Maren Morris", 0.96, 0.84),
("The Bones", "Maren Morris", 0.94, 0.76),
("80s Mercedes", "Maren Morris", 0.96, 0.86),
("Girl", "Maren Morris", 0.92, 0.80),
("I Could Use a Love Song", "Maren Morris", 0.90, 0.72),

("Wagon Wheel", "Darius Rucker", 0.98, 0.86),
("Alright", "Darius Rucker", 0.96, 0.80),
("Come Back Song", "Darius Rucker", 0.94, 0.82),
("Homegrown Honey", "Darius Rucker", 0.96, 0.84),
("If I Told You", "Darius Rucker", 0.90, 0.72),

("Die a Happy Man", "Thomas Rhett", 0.96, 0.74),
("Crash and Burn", "Thomas Rhett", 0.94, 0.82),
("Marry Me", "Thomas Rhett", 0.88, 0.70),
("Life Changes", "Thomas Rhett", 0.96, 0.84),
("Beer Can't Fix", "Thomas Rhett", 0.98, 0.90),

("Take Your Time", "Sam Hunt", 0.92, 0.72),
("House Party", "Sam Hunt", 0.98, 0.92),
("Body Like a Back Road", "Sam Hunt", 0.96, 0.86),
("Break Up in a Small Town", "Sam Hunt", 0.90, 0.78),
("Make You Miss Me", "Sam Hunt", 0.92, 0.74),

("Beautiful Crazy", "Luke Combs", 0.96, 0.72),
("When It Rains It Pours", "Luke Combs", 0.98, 0.88),
("Hurricane", "Luke Combs", 0.92, 0.82),
("She Got the Best of Me", "Luke Combs", 0.90, 0.74),
("Lovin' on You", "Luke Combs", 0.96, 0.86),


("Forever and Ever, Amen", "Randy Travis", 0.96, 0.80),
("Deeper Than the Holler", "Randy Travis", 0.92, 0.74),
("Three Wooden Crosses", "Randy Travis", 0.90, 0.70),
("On the Other Hand", "Randy Travis", 0.88, 0.68),
("Diggin' Up Bones", "Randy Travis", 0.86, 0.76),

("Strawberry Wine", "Deana Carter", 0.92, 0.70),
("Did I Shave My Legs for This?", "Deana Carter", 0.94, 0.78),
("We Danced Anyway", "Deana Carter", 0.96, 0.82),
("Count Me In", "Deana Carter", 0.90, 0.74),
("How Do I Get There", "Deana Carter", 0.88, 0.72),

("Independence Day", "Martina McBride", 0.88, 0.82),
("A Broken Wing", "Martina McBride", 0.84, 0.78),
("Valentine", "Martina McBride", 0.94, 0.72),
("This One's for the Girls", "Martina McBride", 0.96, 0.84),
("Blessed", "Martina McBride", 0.92, 0.76),

("Brand New Man", "Brooks & Dunn", 0.98, 0.90),
("Hard Workin' Man", "Brooks & Dunn", 0.96, 0.88),
("You're Gonna Miss Me When I'm Gone", "Brooks & Dunn", 0.94, 0.84),
("She Used to Be Mine", "Brooks & Dunn", 0.88, 0.72),
("Hillbilly Deluxe", "Brooks & Dunn", 0.98, 0.92),

("Redneck Woman", "Gretchen Wilson", 0.96, 0.90),
("Here for the Party", "Gretchen Wilson", 0.98, 0.92),
("Homewrecker", "Gretchen Wilson", 0.94, 0.88),
("When I Think About Cheatin'", "Gretchen Wilson", 0.88, 0.74),
("All Jacked Up", "Gretchen Wilson", 0.96, 0.94),

("Mud on the Tires", "Brad Paisley", 0.98, 0.88),
("Whiskey Lullaby", "Brad Paisley", 0.84, 0.68),
("Then", "Brad Paisley", 0.94, 0.74),
("She's Everything", "Brad Paisley", 0.96, 0.78),
("Ticks", "Brad Paisley", 0.98, 0.90),

("Springsteen", "Eric Church", 0.94, 0.80),
("Drink in My Hand", "Eric Church", 0.98, 0.92),
("Talladega", "Eric Church", 0.92, 0.74),
("Record Year", "Eric Church", 0.96, 0.86),
("Give Me Back My Hometown", "Eric Church", 0.90, 0.72),

("Drunk on You", "Luke Bryan", 0.98, 0.92),
("Play It Again", "Luke Bryan", 0.96, 0.88),
("Crash My Party", "Luke Bryan", 0.94, 0.84),
("That's My Kind of Night", "Luke Bryan", 0.98, 0.94),
("Strip It Down", "Luke Bryan", 0.92, 0.76),

("Girl Crush", "Little Big Town", 0.88, 0.70),
("Pontoon", "Little Big Town", 0.98, 0.90),
("Boondocks", "Little Big Town", 0.96, 0.86),
("Better Man", "Little Big Town", 0.90, 0.72),
("Tornado", "Little Big Town", 0.94, 0.88),


("Chattahoochee", "Alan Jackson", 0.98, 0.90),
("Remember When", "Alan Jackson", 0.92, 0.70),
("Drive (For Daddy Gene)", "Alan Jackson", 0.94, 0.72),
("It's Five O'Clock Somewhere", "Alan Jackson", 0.98, 0.88),
("Little Bitty", "Alan Jackson", 0.96, 0.84),

("Boot Scootin' Boogie (Live)", "Brooks & Dunn", 0.98, 0.90),
("Neon Moon (Live)", "Brooks & Dunn", 0.90, 0.74),
("My Maria (Live)", "Brooks & Dunn", 0.96, 0.84),
("Only in America (Live)", "Brooks & Dunn", 0.98, 0.88),
("Red Dirt Road (Live)", "Brooks & Dunn", 0.92, 0.76),

("Blue Clear Sky", "George Strait", 0.96, 0.84),
("I Cross My Heart", "George Strait", 0.94, 0.72),
("Ocean Front Property", "George Strait", 0.92, 0.80),
("All My Ex's Live in Texas", "George Strait", 0.98, 0.88),
("Give It Away", "George Strait", 0.90, 0.78),

("You're Still the One", "Shania Twain", 0.96, 0.74),
("Any Man of Mine", "Shania Twain", 0.98, 0.90),
("Man! I Feel Like a Woman!", "Shania Twain", 0.99, 0.94),
("From This Moment On", "Shania Twain", 0.94, 0.72),
("Honey, I'm Home", "Shania Twain", 0.98, 0.92),

("Some Girls Do", "Sawyer Brown", 0.96, 0.88),
("Step That Step", "Sawyer Brown", 0.98, 0.90),
("The Walk", "Sawyer Brown", 0.92, 0.74),
("Six Days on the Road", "Sawyer Brown", 0.94, 0.86),
("This Night Won't Last Forever", "Sawyer Brown", 0.90, 0.72),

("National Working Woman's Holiday", "Sammy Kershaw", 0.96, 0.88),
("She Don't Know She's Beautiful", "Sammy Kershaw", 0.98, 0.90),
("Love of My Life", "Sammy Kershaw", 0.94, 0.74),
("Queen of My Double Wide Trailer", "Sammy Kershaw", 0.98, 0.92),
("Cadillac Style", "Sammy Kershaw", 0.96, 0.86),

("Heads Carolina, Tails California", "Jo Dee Messina", 0.98, 0.90),
("Bye Bye", "Jo Dee Messina", 0.96, 0.88),
("Stand Beside Me", "Jo Dee Messina", 0.94, 0.80),
("Bring On the Rain", "Jo Dee Messina", 0.90, 0.74),
("Lesson in Leavin'", "Jo Dee Messina", 0.92, 0.82),

("Somewhere with You", "Kenny Chesney", 0.90, 0.72),
("Get Along", "Kenny Chesney", 0.96, 0.84),
("American Kids", "Kenny Chesney", 0.98, 0.90),
("Summertime", "Kenny Chesney", 0.98, 0.88),
("No Shoes, No Shirt, No Problems", "Kenny Chesney", 0.99, 0.92),

("Barefoot Blue Jean Night", "Jake Owen", 0.98, 0.92),
("Beachin'", "Jake Owen", 0.96, 0.86),
("Anywhere with You", "Jake Owen", 0.98, 0.90),
("Real Life", "Jake Owen", 0.94, 0.84),
("Down to the Honkytonk", "Jake Owen", 0.98, 0.92),



("Achy Breaky Heart", "Billy Ray Cyrus", 0.98, 0.90),
("Some Gave All", "Billy Ray Cyrus", 0.90, 0.72),
("Busy Man", "Billy Ray Cyrus", 0.94, 0.84),
("Ready, Set, Don't Go", "Billy Ray Cyrus", 0.92, 0.76),
("Trail of Tears", "Billy Ray Cyrus", 0.88, 0.74),

("Daddy's Hands", "Holly Dunn", 0.92, 0.68),
("You Really Had Me Going", "Holly Dunn", 0.94, 0.80),
("Strangers Again", "Holly Dunn", 0.90, 0.72),
("Maybe I Mean Yes", "Holly Dunn", 0.96, 0.82),
("There Goes My Heart Again", "Holly Dunn", 0.88, 0.70),

("Meet in the Middle", "Diamond Rio", 0.96, 0.86),
("One More Day", "Diamond Rio", 0.92, 0.70),
("Beautiful Mess", "Diamond Rio", 0.90, 0.72),
("Love a Little Stronger", "Diamond Rio", 0.94, 0.84),
("How Your Love Makes Me Feel", "Diamond Rio", 0.98, 0.90),

("Watermelon Crawl", "Tracy Byrd", 0.98, 0.92),
("Keeper of the Stars", "Tracy Byrd", 0.94, 0.72),
("The Truth About Men", "Tracy Byrd", 0.96, 0.88),
("Ten Rounds with Jose Cuervo", "Tracy Byrd", 0.98, 0.94),
("Drinkin' Bone", "Tracy Byrd", 0.96, 0.90),

("She's in Love with the Boy", "Trisha Yearwood", 0.96, 0.82),
("How Do I Live", "Trisha Yearwood", 0.94, 0.70),
("Walkaway Joe", "Trisha Yearwood", 0.92, 0.74),
("XXX's and OOO's", "Trisha Yearwood", 0.98, 0.88),
("Perfect Love", "Trisha Yearwood", 0.96, 0.84),

("In a Different Light", "Doug Stone", 0.92, 0.72),
("Why Didn't I Think of That", "Doug Stone", 0.94, 0.84),
("I'd Be Better Off (In a Pine Box)", "Doug Stone", 0.88, 0.68),
("Too Busy Being in Love", "Doug Stone", 0.90, 0.70),
("Come in Out of the Pain", "Doug Stone", 0.86, 0.72),

("Born to Fly", "Sara Evans", 0.98, 0.88),
("Suds in the Bucket", "Sara Evans", 0.96, 0.90),
("A Little Bit Stronger", "Sara Evans", 0.94, 0.76),
("No Place That Far", "Sara Evans", 0.92, 0.72),
("I Could Not Ask for More", "Sara Evans", 0.96, 0.78),

("Alright", "Darius Rucker", 0.96, 0.80),
("Come Back Song", "Darius Rucker", 0.94, 0.82),
("Homegrown Honey", "Darius Rucker", 0.96, 0.84),
("Wagon Wheel (Live)", "Darius Rucker", 0.98, 0.88),
("Southern State of Mind", "Darius Rucker", 0.92, 0.76),

("Buy Me a Boat", "Chris Janson", 0.98, 0.90),
("Fix a Drink", "Chris Janson", 0.98, 0.92),
("Good Vibes", "Chris Janson", 0.99, 0.94),
("Drunk Girl", "Chris Janson", 0.90, 0.74),
("Holdin' Her", "Chris Janson", 0.92, 0.76),



("Where Were You (When the World Stopped Turning)", "Alan Jackson", 0.62, 0.40),
("Remember When", "Alan Jackson", 0.74, 0.42),
("Drive (For Daddy Gene)", "Alan Jackson", 0.78, 0.48),
("Chasin' That Neon Rainbow", "Alan Jackson", 0.80, 0.55),
("Midnight in Montgomery", "Alan Jackson", 0.58, 0.46),

("Go Rest High on That Mountain", "Vince Gill", 0.60, 0.38),
("When I Call Your Name", "Vince Gill", 0.66, 0.44),
("Look at Us", "Vince Gill", 0.82, 0.40),
("Pocket Full of Gold", "Vince Gill", 0.70, 0.48),
("Don't Let Our Love Start Slippin' Away", "Vince Gill", 0.84, 0.60),

("If I Didn't Have You", "Randy Travis", 0.78, 0.48),
("He Walked on Water", "Randy Travis", 0.72, 0.42),
("Is It Still Over?", "Randy Travis", 0.66, 0.50),
("Too Gone Too Long", "Randy Travis", 0.70, 0.52),
("Hard Rock Bottom of Your Heart", "Randy Travis", 0.64, 0.46),

("Neon Moon", "Brooks & Dunn", 0.68, 0.52),
("Believe", "Brooks & Dunn", 0.72, 0.44),
("That Ain't No Way to Go", "Brooks & Dunn", 0.66, 0.56),
("She's Not the Cheatin' Kind", "Brooks & Dunn", 0.70, 0.58),
("Lost and Found", "Brooks & Dunn", 0.74, 0.60),

("The Dance", "Garth Brooks", 0.72, 0.44),
("Unanswered Prayers", "Garth Brooks", 0.82, 0.52),
("Two of a Kind, Workin' on a Full House", "Garth Brooks", 0.88, 0.70),
("Rodeo", "Garth Brooks", 0.84, 0.74),
("Standing Outside the Fire", "Garth Brooks", 0.86, 0.72),

("Independence Day", "Martina McBride", 0.64, 0.62),
("Concrete Angel", "Martina McBride", 0.52, 0.44),
("Wild Angels", "Martina McBride", 0.78, 0.54),
("Love's the Only House", "Martina McBride", 0.82, 0.58),
("Safe in the Arms of Love", "Martina McBride", 0.88, 0.68),

("Something Like That", "Tim McGraw", 0.86, 0.72),
("Just to See You Smile", "Tim McGraw", 0.78, 0.50),
("Don't Take the Girl", "Tim McGraw", 0.70, 0.48),
("I Like It, I Love It", "Tim McGraw", 0.88, 0.70),
("Real Good Man", "Tim McGraw", 0.84, 0.68),

("Strawberry Wine", "Deana Carter", 0.80, 0.50),
("We Danced Anyway", "Deana Carter", 0.86, 0.64),
("How Do I Get There", "Deana Carter", 0.72, 0.48),
("Count Me In", "Deana Carter", 0.82, 0.60),
("Did I Shave My Legs for This?", "Deana Carter", 0.88, 0.66),

("Carrying Your Love with Me", "George Strait", 0.84, 0.58),
("The Chair", "George Strait", 0.80, 0.44),
("Write This Down", "George Strait", 0.88, 0.62),
("Living and Living Well", "George Strait", 0.86, 0.60),
("Give It Away", "George Strait", 0.70, 0.52),

("Whiskey Lullaby", "Brad Paisley", 0.46, 0.40),
("Then", "Brad Paisley", 0.84, 0.50),
("She's Everything", "Brad Paisley", 0.88, 0.56),
("He Didn't Have to Be", "Brad Paisley", 0.82, 0.48),
("Online", "Brad Paisley", 0.90, 0.66),



("Love Story", "Taylor Swift", 0.92, 0.66),
("Teardrops on My Guitar", "Taylor Swift", 0.68, 0.52),
("Our Song", "Taylor Swift", 0.94, 0.74),
("Tim McGraw", "Taylor Swift", 0.78, 0.48),
("Fifteen", "Taylor Swift", 0.72, 0.50),

("Need You Now", "Lady A", 0.70, 0.46),
("American Honey", "Lady A", 0.82, 0.52),
("Just a Kiss", "Lady A", 0.88, 0.60),
("Downtown", "Lady A", 0.92, 0.70),
("Bartender", "Lady A", 0.90, 0.72),

("Humble and Kind", "Tim McGraw", 0.84, 0.46),
("Live Like You Were Dying", "Tim McGraw", 0.80, 0.50),
("My Best Friend", "Tim McGraw", 0.88, 0.60),
("Something Like That", "Tim McGraw", 0.86, 0.68),
("Southern Voice", "Tim McGraw", 0.82, 0.62),

("Somebody Like You", "Keith Urban", 0.92, 0.74),
("You'll Think of Me", "Keith Urban", 0.76, 0.52),
("Blue Ain't Your Color", "Keith Urban", 0.70, 0.44),
("Days Go By", "Keith Urban", 0.94, 0.78),
("Wasted Time", "Keith Urban", 0.90, 0.72),

("Chicken Fried", "Zac Brown Band", 0.92, 0.70),
("Colder Weather", "Zac Brown Band", 0.68, 0.50),
("Free", "Zac Brown Band", 0.84, 0.54),
("Toes", "Zac Brown Band", 0.94, 0.76),
("Homegrown", "Zac Brown Band", 0.90, 0.72),

("Drink a Beer", "Luke Bryan", 0.58, 0.44),
("Play It Again", "Luke Bryan", 0.92, 0.74),
("Crash My Party", "Luke Bryan", 0.88, 0.66),
("Strip It Down", "Luke Bryan", 0.82, 0.54),
("Drunk on You", "Luke Bryan", 0.94, 0.80),

("Beautiful Crazy", "Luke Combs", 0.88, 0.54),
("When It Rains It Pours", "Luke Combs", 0.92, 0.76),
("Hurricane", "Luke Combs", 0.78, 0.68),
("Beer Never Broke My Heart", "Luke Combs", 0.94, 0.82),
("Lovin' on You", "Luke Combs", 0.90, 0.72),

("Body Like a Back Road", "Sam Hunt", 0.90, 0.72),
("Take Your Time", "Sam Hunt", 0.82, 0.56),
("House Party", "Sam Hunt", 0.94, 0.84),
("Break Up in a Small Town", "Sam Hunt", 0.72, 0.60),
("Make You Miss Me", "Sam Hunt", 0.84, 0.58),

("My Church", "Maren Morris", 0.90, 0.70),
("The Bones", "Maren Morris", 0.84, 0.54),
("Girl", "Maren Morris", 0.82, 0.60),
("80s Mercedes", "Maren Morris", 0.92, 0.76),
("Rich", "Maren Morris", 0.88, 0.70),



("Take It Easy", "Eagles", 0.90, 0.66),
("Lyin' Eyes", "Eagles", 0.72, 0.50),
("Peaceful Easy Feeling", "Eagles", 0.84, 0.46),
("Already Gone", "Eagles", 0.88, 0.70),
("Best of My Love", "Eagles", 0.86, 0.54),

("Take Me as I Am", "Faith Hill", 0.84, 0.56),
("This Kiss", "Faith Hill", 0.92, 0.72),
("Breathe", "Faith Hill", 0.88, 0.50),
("There You'll Be", "Faith Hill", 0.82, 0.48),
("Mississippi Girl", "Faith Hill", 0.90, 0.66),

("Austin", "Blake Shelton", 0.76, 0.48),
("God's Country", "Blake Shelton", 0.84, 0.72),
("Honey Bee", "Blake Shelton", 0.92, 0.74),
("Boys 'Round Here", "Blake Shelton", 0.94, 0.80),
("Home", "Blake Shelton", 0.80, 0.44),

("What Hurts the Most", "Rascal Flatts", 0.64, 0.48),
("Bless the Broken Road", "Rascal Flatts", 0.86, 0.52),
("Life Is a Highway", "Rascal Flatts", 0.92, 0.78),
("Mayberry", "Rascal Flatts", 0.84, 0.60),
("Fast Cars and Freedom", "Rascal Flatts", 0.88, 0.70),

("Somebody's Hero", "Jamie O'Neal", 0.82, 0.50),
("There Is No Arizona", "Jamie O'Neal", 0.70, 0.52),
("When I Think About Angels", "Jamie O'Neal", 0.86, 0.62),
("Frantic", "Jamie O'Neal", 0.74, 0.60),
("Trying to Find Atlantis", "Jamie O'Neal", 0.84, 0.64),

("If You're Going Through Hell", "Rodney Atkins", 0.88, 0.68),
("Watching You", "Rodney Atkins", 0.90, 0.64),
("Take a Back Road", "Rodney Atkins", 0.92, 0.72),
("These Are My People", "Rodney Atkins", 0.86, 0.66),
("Farmer's Daughter", "Rodney Atkins", 0.82, 0.58),

("Leave the Pieces", "The Wreckers", 0.72, 0.54),
("Stand Still, Look Pretty", "The Wreckers", 0.84, 0.60),
("My, Oh My", "The Wreckers", 0.80, 0.58),
("Tennessee", "The Wreckers", 0.82, 0.52),
("Lay Me Down", "The Wreckers", 0.78, 0.48),

("She Don't Know She's Beautiful", "Sammy Kershaw", 0.92, 0.74),
("Queen of My Double Wide Trailer", "Sammy Kershaw", 0.94, 0.78),
("Love of My Life", "Sammy Kershaw", 0.84, 0.56),
("Cadillac Style", "Sammy Kershaw", 0.88, 0.70),
("National Working Woman's Holiday", "Sammy Kershaw", 0.90, 0.72),

("Red Ragtop", "Tim McGraw", 0.66, 0.50),
("Everywhere", "Tim McGraw", 0.86, 0.68),
("Where the Green Grass Grows", "Tim McGraw", 0.88, 0.66),
("Something Like That (Live)", "Tim McGraw", 0.90, 0.70),
("Down on the Farm", "Tim McGraw", 0.92, 0.78),


("Tennessee Whiskey", "Chris Stapleton", 0.82, 0.48),
("Broken Halos", "Chris Stapleton", 0.78, 0.62),
("Starting Over", "Chris Stapleton", 0.84, 0.54),
("Fire Away", "Chris Stapleton", 0.66, 0.50),
("Millionaire", "Chris Stapleton", 0.88, 0.60),

("Traveller", "Chris Stapleton", 0.80, 0.52),
("Nobody to Blame", "Chris Stapleton", 0.74, 0.60),
("Maggie's Song", "Chris Stapleton", 0.62, 0.42),
("Arkansas", "Chris Stapleton", 0.86, 0.68),
("Parachute", "Chris Stapleton", 0.84, 0.66),

("Colder Weather (Live)", "Zac Brown Band", 0.70, 0.50),
("Chicken Fried (Live)", "Zac Brown Band", 0.90, 0.70),
("Free (Live)", "Zac Brown Band", 0.86, 0.56),
("Toes (Live)", "Zac Brown Band", 0.92, 0.74),
("Highway 20 Ride", "Zac Brown Band", 0.72, 0.48),

("You Should Probably Leave", "Chris Stapleton", 0.68, 0.46),
("Joy of My Life", "Chris Stapleton", 0.84, 0.54),
("White Horse", "Chris Stapleton", 0.74, 0.62),
("Devil Always Made Me Think Twice", "Chris Stapleton", 0.70, 0.58),
("Second One to Know", "Chris Stapleton", 0.76, 0.64),

("Hometown Girl", "Josh Turner", 0.88, 0.62),
("Long Black Train", "Josh Turner", 0.72, 0.48),
("Your Man", "Josh Turner", 0.84, 0.54),
("Time Is Love", "Josh Turner", 0.90, 0.66),
("Why Don't We Just Dance", "Josh Turner", 0.92, 0.70),

("The House That Built Me", "Miranda Lambert", 0.74, 0.44),
("Mama's Broken Heart", "Miranda Lambert", 0.86, 0.68),
("Bluebird", "Miranda Lambert", 0.82, 0.54),
("Automatic", "Miranda Lambert", 0.78, 0.52),
("Tin Man", "Miranda Lambert", 0.64, 0.42),

("My Wish", "Rascal Flatts", 0.86, 0.50),
("Here Comes Goodbye", "Rascal Flatts", 0.64, 0.48),
("Stand", "Rascal Flatts", 0.88, 0.62),
("Take Me There", "Rascal Flatts", 0.90, 0.68),
("Rewind", "Rascal Flatts", 0.84, 0.60),

("The Good Stuff", "Kenny Chesney", 0.82, 0.52),
("Anything but Mine", "Kenny Chesney", 0.80, 0.50),
("Beer in Mexico", "Kenny Chesney", 0.86, 0.68),
("When the Sun Goes Down", "Kenny Chesney", 0.90, 0.74),
("Somewhere with You", "Kenny Chesney", 0.78, 0.52),



("Follow Your Arrow", "Kacey Musgraves", 0.88, 0.62),
("Merry Go 'Round", "Kacey Musgraves", 0.72, 0.50),
("Biscuits", "Kacey Musgraves", 0.86, 0.64),
("Slow Burn", "Kacey Musgraves", 0.80, 0.46),
("Butterflies", "Kacey Musgraves", 0.90, 0.58),

("Dirt Road Anthem", "Jason Aldean", 0.84, 0.70),
("Big Green Tractor", "Jason Aldean", 0.88, 0.62),
("She's Country", "Jason Aldean", 0.90, 0.74),
("Fly Over States", "Jason Aldean", 0.82, 0.60),
("Night Train", "Jason Aldean", 0.86, 0.68),

("Take It from Me", "Jordan Davis", 0.86, 0.66),
("Singles You Up", "Jordan Davis", 0.88, 0.70),
("Slow Dance in a Parking Lot", "Jordan Davis", 0.82, 0.54),
("Almost Maybes", "Jordan Davis", 0.80, 0.58),
("Buy Dirt", "Jordan Davis", 0.90, 0.52),

("Somebody Else Will", "Justin Moore", 0.84, 0.64),
("Small Town USA", "Justin Moore", 0.88, 0.66),
("Til My Last Day", "Justin Moore", 0.90, 0.60),
("Bait a Hook", "Justin Moore", 0.92, 0.72),
("Point at You", "Justin Moore", 0.86, 0.68),

("You Should Be Here", "Cole Swindell", 0.70, 0.48),
("Middle of a Memory", "Cole Swindell", 0.82, 0.56),
("Chillin' It", "Cole Swindell", 0.90, 0.70),
("Love You Too Late", "Cole Swindell", 0.74, 0.54),
("Break Up in the End", "Cole Swindell", 0.66, 0.50),

("I Hope You're Happy Now", "Carly Pearce", 0.72, 0.58),
("Every Little Thing", "Carly Pearce", 0.68, 0.50),
("Next Girl", "Carly Pearce", 0.82, 0.66),
("Hide the Wine", "Carly Pearce", 0.86, 0.72),
("What He Didn't Do", "Carly Pearce", 0.64, 0.46),

("Lady May", "Tyler Childers", 0.84, 0.46),
("Whitehouse Road", "Tyler Childers", 0.78, 0.66),
("All Your'n", "Tyler Childers", 0.88, 0.58),
("Feathered Indians", "Tyler Childers", 0.86, 0.62),
("House Fire", "Tyler Childers", 0.90, 0.74),

("Burning House", "Cam", 0.66, 0.44),
("Diane", "Cam", 0.70, 0.52),
("Till There's Nothing Left", "Cam", 0.84, 0.60),
("Mayday", "Cam", 0.72, 0.56),
("Country Ain't Never Been Pretty", "Cam", 0.82, 0.58),

("More Hearts Than Mine", "Ingrid Andress", 0.76, 0.50),
("Lady Like", "Ingrid Andress", 0.84, 0.64),
("Wishful Drinking", "Ingrid Andress", 0.78, 0.58),
("Seeing Someone Else", "Ingrid Andress", 0.72, 0.54),
("Good Person", "Ingrid Andress", 0.80, 0.56)


]

# ---------------- YOUTUBE SEARCH SETTINGS ----------------
ydl_opts = {
    "quiet": True,
    "skip_download": True,
    "noplaylist": True,
    "extract_flat": True,
    "ignoreerrors": True
}

# ---------------- FUNCTION TO FETCH LINK ----------------
def get_link(song, artist):
    try:
        query = f"ytsearch1:{song} {artist} official video"
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)

            if not info or "entries" not in info or not info["entries"]:
                return ""

            video_id = info["entries"][0].get("id")
            if video_id:
                return f"https://www.youtube.com/watch?v={video_id}"

            return ""

    except Exception:
        return ""


# ---------------- WRITE FINAL DATASET FILE ----------------
output_file = "country_links_filled.py"

with open(output_file, "w", encoding="utf-8") as f:
    for song, artist, valence, arousal in songs:
        link = get_link(song, artist)

        line = f'("{song}", "{artist}", "English", {valence}, {arousal}, "Country", ["{link}"]),\n'
        f.write(line)

print("✅ Country dataset with YouTube links generated ->", output_file)
