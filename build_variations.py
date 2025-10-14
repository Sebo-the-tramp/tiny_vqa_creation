import json
import re

with open('simpler.json') as f:
    data = json.load(f)

variations = {
    "F_VISIBILITY_COUNTING_TOTAL_VISIBLE": [
        "How many objects stay within view for the entire sequence?",
        "What is the total number of objects visible from start to finish?",
        "Can you count the objects that never leave the frame during the sequence?",
        "How many items remain in sight throughout the whole sequence?",
        "How many objects can be seen for the full duration of the sequence?",
        "How many bodies stay observable the entire time?",
        "How many objects remain visible across every frame of the sequence?",
        "What is the count of objects that stay visible during the complete sequence?",
        "How many objects keep appearing on-screen for the whole sequence?",
        "What is the tally of objects that are visible the whole way through?",
    ],
    "F_VISIBILITY_COUNTING_EVER_VISIBLE": [
        "How many distinct objects come into view at some point?",
        "What is the number of unique objects that appear on-screen at any moment?",
        "How many separate objects ever become visible during the sequence?",
        "Can you tally the unique objects that show up at any time?",
        "How many different objects are seen at any stage of the sequence?",
        "What count of unique objects pops into view over the sequence?",
        "How many individual objects become visible at least once?",
        "How many unique items make an appearance during the sequence?",
        "What is the total of distinct objects that show themselves at any time?",
        "How many different objects ever enter the frame visibly?",
    ],
    "F_VISIBILITY_COUNTING_OCCLUDED": [
        "How many objects end up fully occluded at some moment?",
        "What is the count of objects that get totally blocked from view at any time?",
        "How many objects become entirely hidden during the sequence?",
        "How many items are completely occluded at any stage?",
        "How many objects experience a moment of total occlusion?",
        "Can you count the objects that get fully covered at any point?",
        "How many objects are ever entirely obscured from sight?",
        "What number of objects becomes completely out of view at any time?",
        "How many objects are totally occluded during the sequence?",
        "How many objects get completely masked from the camera at some point?",
    ],
    "F_VISIBILITY_ATTRIBUTE_FIRST_APPEAR": [
        "Which object is the first to come into view?",
        "Identify the object that appears earliest?",
        "Which item shows up before all the others?",
        "Which object makes the initial appearance?",
        "Which object is seen first in the sequence?",
        "Which object enters the viewer's sightline first?",
        "Point out the object that becomes visible ahead of the rest?",
        "Which object is observed first on-screen?",
        "Name the object that appears first in the sequence?",
        "Which object shows itself before any other?",
    ],
    "F_VISIBILITY_ATTRIBUTE_LAST_DISAPPEAR": [
        "Which object disappears from sight last?",
        "Identify the object that lingers the longest before vanishing?",
        "Which item is still visible after all the others are gone?",
        "Which object exits the frame at the very end?",
        "Which object is seen last before the scene clears?",
        "Which object remains in view the longest before disappearing?",
        "Point out the object that is the final one to leave sight?",
        "Which object is the last to drop out of view?",
        "Name the object that stays visible until the end?",
        "Which object vanishes only after every other has gone?",
    ],
    "F_VISIBILITY_ATTRIBUTE_LONGEST_OCCLUDER": [
        "Which object blocks another for the greatest duration?",
        "Identify the object that produces the lengthiest occlusion?",
        "Which item keeps another object hidden for the longest time?",
        "Which object is responsible for the longest-lasting occlusion event?",
        "Which object causes the most extended blockage of another?",
        "Which object maintains an occlusion over another object the longest?",
        "Point to the object that occludes another for the greatest span?",
        "Which object holds another out of view for the longest period?",
        "Name the object that delivers the longest occluding effect?",
        "Which object creates the longest sustained occlusion of a peer?",
    ],
    "F_VISIBILITY_ESTIMATION_VISIBLE_DURATION": [
        "About how long does the object remain visible?",
        "What is your estimate of the time the object stays in view?",
        "Roughly how much time is the object visible?",
        "Can you approximate the duration the object is seen?",
        "What is the visibility duration of the object, approximately?",
        "Give an estimate of how long the object stays observable?",
        "Roughly for how long does the object stay in sight?",
        "What would you say is the object's visible time span?",
        "Approximately how long is the object visible on-screen?",
        "Estimate the length of time the object remains in view?",
    ],
    "F_VISIBILITY_ATTRIBUTE_ENTRY_SIDE": [
        "Which side of the frame does the object enter from initially?",
        "Identify the side where the object makes its first entrance?",
        "Where on the frame boundary does the object appear first?",
        "From what side does the object come into the frame?",
        "Which edge of the frame does the object cross first?",
        "On which side does the object first show up in the frame?",
        "Point out the frame side where the object enters to begin with?",
        "From which direction along the frame does the object first arrive?",
        "Which boundary of the frame does the object enter through at the start?",
        "Where does the object first step into the frame?",
    ],
    "F_TEMPORAL_ATTRIBUTE_FIRST_CONTACT": [
        "Which object touches the ground before the others?",
        "Identify the object that hits the ground first?",
        "Which item makes the earliest ground contact?",
        "Which object reaches the ground ahead of the rest?",
        "Which object is the first to land?",
        "Point to the object that contacts the surface first?",
        "Which object makes initial contact with the ground?",
        "Which object lands first in the sequence?",
        "Name the object that hits the ground earliest?",
        "Which object is observed contacting the ground first?",
    ],
    "F_TEMPORAL_ATTRIBUTE_LAST_EXIT": [
        "Which object exits the frame last?",
        "Identify the object that is the final one to leave the frame?",
        "Which item remains in frame until the very end?",
        "Which object stays on-screen the longest before exiting?",
        "Which object is seen leaving the frame after all others?",
        "Point out the object that is last to depart the frame?",
        "Which object makes the final exit from the frame?",
        "Name the object that lags behind and leaves last?",
        "Which object holds in the frame the longest before leaving?",
        "Which object is the last to move out of the frame?",
    ],
    "F_TEMPORAL_COUNTING_SIMULTANEOUS_MOVE": [
        "How many objects move at the same time when the target starts to fall?",
        "What number of objects move together as the target object begins to fall?",
        "Can you count the objects that move simultaneously when the target starts falling?",
        "How many items move in unison when the target object begins its fall?",
        "How many objects are moving at the moment the target object starts to drop?",
        "How many objects move concurrently as soon as the target begins to fall?",
        "How many other objects move right when the target object starts descending?",
        "What is the tally of objects that move in sync with the target's initial fall?",
        "How many objects initiate motion at the same instant the target begins to fall?",
        "How many objects respond by moving when the target object starts to fall?",
    ],
    "F_TEMPORAL_COUNTING_PHASES": [
        "How many separate motion phases does the target object experience?",
        "What number of distinct motion phases does the target object undergo?",
        "Can you count the unique motion phases the target object passes through?",
        "How many different motion phases occur for the target object?",
        "How many motion phases does the target object transition through?",
        "How many discrete motion phases does the target object exhibit?",
        "What is the count of motion phases the target object moves through?",
        "How many stages of motion does the target object go through?",
        "How many distinct phases characterize the target object's motion?",
        "How many motion segments does the target object pass through?",
    ],
    "F_TEMPORAL_ESTIMATION_INTERVAL_COLLISIONS": [
        "About how much time separates the first and second collisions?",
        "What is your estimate of the interval between the first and second collisions?",
        "Roughly how long is it between the first collision and the second?",
        "Can you approximate the time gap between the first and second collisions?",
        "What is the time span between the first and second collisions, approximately?",
        "How much time passes between the first and second collisions?",
        "What would you say is the elapsed time between the first and second collisions?",
        "Estimate the interval separating the first collision from the second?",
        "Approximately how long after the first collision does the second occur?",
        "What is the rough timing between the initial and follow-up collisions?",
    ],
    "F_TEMPORAL_ESTIMATION_STATIC_DURATION": [
        "How long does the object stay still before it moves?",
        "What is the duration of the object's stationary period before motion?",
        "Approximately how long does the object remain at rest before moving?",
        "Can you estimate the time the object stays stationary prior to moving?",
        "How much time passes while the object is stationary before it starts moving?",
        "What length of time does the object sit still before moving?",
        "For how long does the object remain motionless before it begins to move?",
        "How long is the object stationary before it transitions into motion?",
        "Estimate the stationary interval the object spends before moving?",
        "What is the time span the object stays still before it moves?",
    ],
    "F_TEMPORAL_ATTRIBUTE_PEAK_SPEED_INTERVAL": [
        "In what time interval does the object achieve its maximum speed?",
        "Identify when the object reaches its peak speed?",
        "Which time window contains the object's maximum speed?",
        "When does the object hit its highest speed?",
        "Over which interval does the object get to its top speed?",
        "During what segment of time does the object reach maximum speed?",
        "Point out the time interval where the object is moving at its fastest?",
        "Which stretch of time sees the object achieve peak speed?",
        "When exactly does the object attain its maximum speed?",
        "Which interval marks the object's highest speed?",
    ],
    "F_KINEMATICS_COUNTING_LINEAR_MOTION": [
        "How many objects experience prolonged linear motion?",
        "What number of objects move in a sustained straight line?",
        "Can you count the objects that maintain linear motion?",
        "How many items exhibit sustained straight-line motion?",
        "How many objects keep moving linearly over time?",
        "How many objects travel in an extended straight path?",
        "How many bodies persist in linear motion?",
        "What is the count of objects that continue in linear motion?",
        "How many objects follow a sustained straight trajectory?",
        "How many objects remain in straight-line motion throughout?",
    ],
    "F_KINEMATICS_COUNTING_ROTATION": [
        "How many objects undergo rotation about some axis during the sequence?",
        "What number of objects spin around an axis at any point?",
        "Can you count the objects that rotate around an axis in the sequence?",
        "How many items exhibit rotational motion about any axis?",
        "How many objects end up rotating around an axis during the sequence?",
        "How many bodies perform a rotation about some axis?",
        "How many objects show axial rotation in the sequence?",
        "What is the tally of objects that spin around an axis?",
        "How many objects engage in rotation about any axis while the sequence runs?",
        "How many objects turn around an axis during the sequence?",
    ],
    "F_KINEMATICS_ATTRIBUTE_LARGEST_DISPLACEMENT": [
        "Which object undergoes the greatest displacement?",
        "Identify the object that travels the farthest distance?",
        "Which item shows the largest overall displacement?",
        "Which object ends up with the biggest displacement?",
        "Which object is displaced the most?",
        "Which object covers the greatest displacement?",
        "Point out the object that moves the farthest from its start?",
        "Which object registers the largest displacement?",
        "Name the object that experiences the maximum displacement?",
        "Which object has the most significant displacement?",
    ],
    "F_KINEMATICS_ATTRIBUTE_MAX_ANGULAR_SPEED": [
        "Which object rotates at the greatest angular speed?",
        "Identify the object spinning the fastest?",
        "Which item has the highest rate of spin?",
        "Which object achieves the maximum angular velocity?",
        "Which object is turning with the fastest angular speed?",
        "Which object exhibits the highest angular rate?",
        "Point to the object that spins the quickest?",
        "Which object shows the peak angular speed?",
        "Name the object with the greatest angular velocity?",
        "Which object rotates faster than the rest?",
    ],
    "F_KINEMATICS_ATTRIBUTE_PATH_SHAPE": [
        "How would you describe the general shape of the target object's trajectory?",
        "What shape best characterizes the target object's path?",
        "Which overall shape does the target object's trajectory resemble?",
        "What is the target object's trajectory shaped like?",
        "Describe the overall form of the target object's trajectory?",
        "What general trajectory shape does the target object follow?",
        "Which shape captures the target object's path through space?",
        "What would you say is the target object's trajectory shape?",
        "Identify the general shape taken by the target object's trajectory?",
        "What overall path shape does the target object trace out?",
    ],
    "F_KINEMATICS_ESTIMATION_DISPLACEMENT_TARGET": [
        "About how large is the target object's displacement?",
        "What estimate would you give for the target object's displacement?",
        "Approximately how far is the target object displaced?",
        "Can you approximate the displacement of the target object?",
        "What is the target object's displacement, roughly speaking?",
        "Give an estimate of the target object's overall displacement?",
        "What would you say is the displacement experienced by the target object?",
        "Estimate the magnitude of the target object's displacement?",
        "Approximately what displacement does the target object undergo?",
        "What is your estimate of the target object's displacement?",
    ],
    "F_KINEMATICS_ESTIMATION_ANGULAR_DISPLACEMENT": [
        "About how large is the object's angular displacement?",
        "What estimate would you give for the object's angular displacement?",
        "Approximately how much does the object rotate?",
        "Can you approximate the object's total angular displacement?",
        "What is the object's angular displacement, roughly?",
        "Give an estimate of the object's angular displacement?",
        "What would you say is the angular displacement the object experiences?",
        "Estimate the magnitude of the object's angular displacement?",
        "Approximately what angular displacement does the object undergo?",
        "What is your estimate of the object's overall angular displacement?",
    ],
    "F_KINEMATICS_ESTIMATION_PEAK_ACCELERATION": [
        "About how large is the target object's peak acceleration?",
        "What estimate would you give for the target object's maximum acceleration?",
        "Approximately what is the target object's peak acceleration?",
        "Can you approximate the highest acceleration the target object reaches?",
        "What is the target object's peak acceleration, roughly?",
        "Give an estimate of the maximum acceleration experienced by the target object?",
        "What would you say is the target object's top acceleration?",
        "Estimate the magnitude of the target object's peak acceleration?",
        "Approximately how great is the target object's maximum acceleration?",
        "What is your estimate of the target object's peak acceleration?",
    ],
    "F_SPATIAL_ATTRIBUTE_DIRECTLY_ABOVE": [
        "Which object sits directly above the target at the beginning?",
        "Identify the object positioned directly above the target initially?",
        "Which item is located straight above the target at the start?",
        "Which object is found directly over the target when the sequence begins?",
        "Which object hovers directly above the target at the outset?",
        "Point out the object that is directly above the target in the first frame?",
        "Which object lies right above the target at the start?",
        "Which object occupies the spot directly above the target at the beginning?",
        "Name the object directly above the target at the start?",
        "Which object is positioned over the target right at the start?",
    ],
    "F_SPATIAL_ATTRIBUTE_IN_FRONT": [
        "Which object stands in front of the target from the camera's viewpoint?",
        "Identify the object located ahead of the target relative to the camera?",
        "Which item is positioned in front of the target with respect to the camera?",
        "Which object lies between the camera and the target?",
        "Which object is situated before the target as seen by the camera?",
        "Point to the object that sits in front of the target from the camera's perspective?",
        "Which object appears in front of the target from the camera angle?",
        "Name the object occupying the space in front of the target relative to the camera?",
        "Which object is directly in front of the target when viewed from the camera?",
        "Which object is nearest the camera ahead of the target?",
    ],
    "F_SPATIAL_ATTRIBUTE_CLOSEST_FINAL": [
        "Which object ends up closest to the target in the last frame?",
        "Identify the object nearest the target at the final frame?",
        "Which item is closest to the target when the sequence ends?",
        "Which object lies closest to the target in the closing frame?",
        "Which object is nearest to the target at the end?",
        "Point out the object closest to the target in the final frame?",
        "Which object finishes closest to the target in the last frame?",
        "Name the object located nearest the target in the final frame?",
        "Which object is most proximate to the target in the concluding frame?",
        "Which object remains closest to the target at the final moment?",
    ],
    "F_SPATIAL_COUNTING_LEFT_OF_TARGET": [
        "How many objects sit to the left of the target object?",
        "What number of objects are located left of the target object?",
        "Can you count the objects placed to the target object's left side?",
        "How many items lie on the left side of the target object?",
        "How many objects stand to the left of the target object?",
        "How many objects end up positioned to the target object's left?",
        "What is the count of objects that are on the left of the target object?",
        "How many objects occupy positions left of the target object?",
        "How many bodies are situated to the left-hand side of the target object?",
        "How many objects appear on the target object's left?",
    ],
    "F_SPATIAL_COUNTING_WITHIN_DISTANCE": [
        "How many objects lie within one meter of the target object's center?",
        "What number of objects are found inside a meter of the target object's center?",
        "Can you count the objects located within one meter of the target object's center?",
        "How many items fall within one meter of the target object's center?",
        "How many objects stand within a meter of the target object's center?",
        "How many objects are positioned no more than one meter from the target object's center?",
        "What is the tally of objects within a one-meter radius of the target object's center?",
        "How many objects sit inside one meter of the target object's center point?",
        "How many bodies remain within one meter of the target object's center?",
        "How many objects stay closer than one meter to the target object's center?",
    ],
    "F_SPATIAL_ESTIMATION_DISTANCE_PAIR": [
        "About how far apart are the target object and the reference object?",
        "What is your estimate of the distance between the target object and the reference object?",
        "Approximately what separation is there between the target and reference objects?",
        "Can you approximate the distance between the target object and the reference object?",
        "What is the distance between the target and reference objects, roughly?",
        "Give an estimate of the spacing between the target object and the reference object?",
        "What would you say is the distance between the target object and the reference object?",
        "Estimate the distance separating the target object from the reference object?",
        "Approximately how far is the target object from the reference object?",
        "What is the rough distance between the target object and the reference object?",
    ],
    "F_SPATIAL_ESTIMATION_HEIGHT_ABOVE_GROUND": [
        "About how high is the target object above the ground?",
        "What is your estimate of the target object's height above ground level?",
        "Approximately how far above the ground is the target object?",
        "Can you approximate the height of the target object above the ground?",
        "What is the target object's ground clearance, roughly?",
        "Give an estimate of how high the target object sits above the ground?",
        "What would you say is the target object's height from the ground?",
        "Estimate the elevation of the target object above the ground?",
        "Approximately what is the target object's height over the ground?",
        "What is the rough height of the target object relative to the ground?",
    ],
    "F_DEFORMATION_COUNTING_PERMANENT": [
        "How many objects are still deformed when the sequence ends?",
        "What number of objects stay deformed by the end of the sequence?",
        "How many items remain deformed at the sequence's conclusion?",
        "Can you count the objects that end up deformed when the sequence finishes?",
        "How many objects are left deformed at the end?",
        "How many objects stay in a deformed state after the sequence?",
        "How many bodies remain deformed once the sequence is over?",
        "How many objects finish the sequence still deformed?",
        "What is the count of objects that remain deformed by the closing frame?",
        "How many objects are deformed at the conclusion of the sequence?",
    ],
    "F_DEFORMATION_COUNTING_BENDING": [
        "How many objects bend but do not break?",
        "What number of objects flex without snapping?",
        "Can you count the objects that bend yet stay intact?",
        "How many items bend without sustaining a break?",
        "How many objects curve without breaking apart?",
        "How many objects deform by bending while remaining unbroken?",
        "How many bodies bend and avoid breaking?",
        "How many objects undergo bending without fracture?",
        "What is the tally of objects that bend without breaking?",
        "How many objects flex without losing integrity?",
    ],
    "F_DEFORMATION_ATTRIBUTE_TYPE": [
        "Which type of deformation affects the object?",
        "What kind of deformation does the object experience?",
        "Identify the deformation type the object undergoes?",
        "What deformation classification applies to the object?",
        "How would you describe the deformation the object exhibits?",
        "What sort of deformation happens to the object?",
        "Which deformation style does the object display?",
        "What is the nature of the deformation the object undergoes?",
        "What deformation mode is seen in the object?",
        "What form of deformation does the object experience?",
    ],
    "F_DEFORMATION_ATTRIBUTE_MAX": [
        "Which object exhibits the most deformation?",
        "Identify the object displaying the largest deformation?",
        "Which item shows the greatest level of deformation?",
        "Which object undergoes deformation the most severely?",
        "Which object demonstrates the maximum deformation?",
        "Which object experiences the highest degree of deformation?",
        "Point out the object with the most pronounced deformation?",
        "Which object reveals the greatest deformation?",
        "Name the object that deforms the most?",
        "Which object suffers the largest deformation?",
    ],
    "F_DEFORMATION_ATTRIBUTE_TRIGGER": [
        "Which event initiates the object's deformation?",
        "Identify the event that causes the object to deform?",
        "What occurrence triggers the object's deformation?",
        "Which event sets off the deformation in the object?",
        "Which event leads to the object's deformation?",
        "What action causes the object to begin deforming?",
        "Which event brings about the object's deformation?",
        "Point to the event responsible for the object's deformation?",
        "Which incident triggers the object to deform?",
        "Which event starts the deformation process in the object?",
    ],
    "F_DEFORMATION_ESTIMATION_DURATION": [
        "How long does the deformation last?",
        "What is the duration of the deformation?",
        "Approximately how long does the deformation persist?",
        "Can you estimate the period that the deformation remains?",
        "How much time passes while the deformation is present?",
        "For what length of time does the deformation continue?",
        "How long is the deformation maintained?",
        "Estimate how long the deformation stays in effect?",
        "What time span does the deformation cover?",
        "How long does the object stay deformed?",
    ],
    "F_DEFORMATION_ESTIMATION_REQUIRED_FORCE": [
        "About how much force is needed to create the observed deformation?",
        "What is your estimate of the force required for the observed deformation?",
        "Approximately what force produces the observed deformation?",
        "Can you approximate the force necessary to cause the observed deformation?",
        "What force level is required to generate the observed deformation, roughly?",
        "Give an estimate of the force needed to produce the observed deformation?",
        "What would you say is the force required to make the observed deformation?",
        "Estimate the magnitude of force needed for the observed deformation?",
        "Approximately how much force results in the observed deformation?",
        "What is the rough force required to achieve the observed deformation?",
    ],
    "F_MASS_COUNTING_HEAVIER_THAN_REFERENCE": [
        "How many objects outweigh the reference object?",
        "What number of objects are heavier than the reference?",
        "Can you count the objects that have more mass than the reference object?",
        "How many items are heavier than the reference object?",
        "How many objects tip the scale above the reference object?",
        "How many objects possess greater mass than the reference?",
        "How many bodies are heavier compared with the reference object?",
        "What is the count of objects heavier than the reference object?",
        "How many objects exceed the reference object in mass?",
        "How many objects come in heavier than the reference object?",
    ],
    "F_MASS_COUNTING_LIGHTER_THAN_REFERENCE": [
        "How many objects weigh less than the reference object?",
        "What number of objects are lighter than the reference?",
        "Can you count the objects that are lighter than the reference object?",
        "How many items carry less mass than the reference object?",
        "How many objects fall below the reference object in weight?",
        "How many objects are lighter compared with the reference?",
        "What is the count of objects lighter than the reference object?",
        "How many bodies come in lighter than the reference object?",
        "How many objects have less mass than the reference object?",
        "How many objects end up lighter than the reference object?",
    ],
    "F_MASS_ATTRIBUTE_HEAVIEST": [
        "Which object seems to have the largest mass?",
        "Identify the object that looks heaviest?",
        "Which item appears to possess the greatest mass?",
        "Which object looks like it weighs the most?",
        "Which object seems to be the heaviest present?",
        "Point out the object that appears to have the most mass?",
        "Which object gives the impression of having the highest mass?",
        "Name the object that appears heaviest?",
        "Which object seems to carry the most weight?",
        "Which object looks to have the greatest mass?",
    ],
    "F_MASS_ATTRIBUTE_LIGHTEST": [
        "Which object seems to have the least mass?",
        "Identify the object that looks lightest?",
        "Which item appears to possess the smallest mass?",
        "Which object looks like it weighs the least?",
        "Which object seems to be the lightest present?",
        "Point out the object that appears to have the least mass?",
        "Which object gives the impression of having the lowest mass?",
        "Name the object that appears lightest?",
        "Which object seems to carry the least weight?",
        "Which object looks to have the smallest mass?",
    ],
    "F_MASS_ESTIMATION_TARGET": [
        "About how much does the target object weigh?",
        "What is your estimate of the target object's mass?",
        "Approximately what mass does the target object have?",
        "Can you approximate the mass of the target object?",
        "What is the target object's mass, roughly speaking?",
        "Give an estimate of the target object's mass?",
        "What would you say is the target object's mass?",
        "Estimate the mass of the target object?",
        "Approximately how heavy is the target object?",
        "What is the rough mass of the target object?",
    ],
    "F_VOLUME_COUNTING_LARGER_THAN_REFERENCE": [
        "How many objects occupy more volume than the reference object?",
        "What number of objects are larger in volume than the reference?",
        "Can you count the objects that have greater volume than the reference object?",
        "How many items possess a volume exceeding the reference object?",
        "How many objects take up more space than the reference object?",
        "How many objects are bigger in volume compared with the reference?",
        "What is the count of objects whose volume surpasses the reference object?",
        "How many bodies present a larger volume than the reference object?",
        "How many objects measure greater in volume than the reference object?",
        "How many objects feature more volume than the reference object?",
    ],
    "F_VOLUME_ATTRIBUTE_MOST_VOLUME": [
        "Which object takes up the most volume?",
        "Identify the object that occupies the largest volume?",
        "Which item fills the greatest amount of space?",
        "Which object has the largest volume?",
        "Which object claims the most space?",
        "Point out the object with the greatest volume?",
        "Which object covers the most volume?",
        "Name the object that occupies the most space?",
        "Which object has the biggest volume?",
        "Which object appears to take up the most volume?",
    ],
    "F_VOLUME_ESTIMATION_TARGET": [
        "About how much volume does the target object occupy?",
        "What is your estimate of the target object's volume?",
        "Approximately what volume does the target object have?",
        "Can you approximate the volume of the target object?",
        "What is the target object's volume, roughly?",
        "Give an estimate of the target object's volume?",
        "What would you say is the target object's volume?",
        "Estimate the volume presented by the target object?",
        "Approximately how much space does the target object occupy?",
        "What is the rough volume of the target object?",
    ],
    "F_COLLISION_COUNTING_TOTAL_EVENTS": [
        "How many collisions happen during the sequence?",
        "What number of collisions take place over the sequence?",
        "Can you count the collisions that occur in the sequence?",
        "How many collision events are recorded during the sequence?",
        "How many impacts occur throughout the sequence?",
        "What is the tally of collisions during the sequence?",
        "How many collisions do you observe in the sequence?",
        "How many collision incidents are there in the sequence?",
        "How many collisions happen across the sequence?",
        "How many collision events unfold during the sequence?",
    ],
    "F_COLLISION_COUNTING_TARGET_CONTACTS": [
        "How many times does the target object make contact with another object?",
        "What number of collisions involve the target object hitting something else?",
        "Can you count the instances where the target object collides with another object?",
        "How many separate times does the target object collide with another object?",
        "How often does the target object run into another object?",
        "How many collision events involve the target object striking another object?",
        "How many times does the target object end up colliding with something else?",
        "What is the count of collisions between the target object and other objects?",
        "How many moments show the target object colliding with another object?",
        "How many times does the target object impact another object?",
    ],
    "F_COLLISION_ATTRIBUTE_FIRST_INITIATOR": [
        "Which object starts the first collision?",
        "Identify the object that triggers the first collision?",
        "Which item initiates the initial collision?",
        "Which object is responsible for beginning the first collision?",
        "Which object sets off the first collision?",
        "Point out the object that causes the first collision to occur?",
        "Which object kicks off the first collision event?",
        "Which object is the instigator of the first collision?",
        "Name the object that initiates the very first collision?",
        "Which object leads into the first collision?",
    ],
    "F_COLLISION_ATTRIBUTE_LAST_CONTACT": [
        "Which objects participate in the final collision?",
        "Identify the objects that take part in the last collision?",
        "Which items are involved in the closing collision?",
        "Which objects collide in the final impact?",
        "Which objects make contact in the final collision event?",
        "Point out the objects engaged in the last collision?",
        "Which pair of objects features in the final collision?",
        "Which objects are part of the ultimate collision?",
        "Name the objects that collide at the end?",
        "Which objects end up in the final collision?",
    ],
    "F_COLLISION_ATTRIBUTE_OUTCOME": [
        "What happens right after the primary collision?",
        "What is the immediate result of the primary collision?",
        "Describe the outcome that follows the main collision?",
        "What occurs immediately as a consequence of the primary collision?",
        "What is produced instantly by the primary collision?",
        "What immediate effect does the primary collision create?",
        "What takes place at once after the primary collision?",
        "What is the direct outcome of the primary collision?",
        "What change happens immediately due to the primary collision?",
        "What is observed right after the primary collision?",
    ],
    "F_COLLISION_ESTIMATION_IMPACT_SPEED": [
        "About how fast are the bodies moving relative to each other at impact?",
        "What is your estimate of the relative speed at collision?",
        "Approximately what relative speed occurs at the moment of collision?",
        "Can you approximate the relative speed when the collision happens?",
        "What is the relative speed at impact, roughly?",
        "Give an estimate of the relative velocity at the moment of collision?",
        "What would you say is the relative speed when they collide?",
        "Estimate the relative speed present at the instant of collision?",
        "Approximately how great is the relative speed during the collision?",
        "What is the rough relative speed at the collision moment?",
    ],
    "F_COLLISION_ESTIMATION_ENERGY_LOSS": [
        "What portion of kinetic energy is lost during the collision?",
        "How much of the kinetic energy is lost in the collision?",
        "What fraction of the kinetic energy disappears in the collision?",
        "Can you estimate the share of kinetic energy lost in the collision?",
        "What percentage of kinetic energy is shed by the collision?",
        "What part of the kinetic energy is dissipated in the collision?",
        "How much kinetic energy is consumed during the collision?",
        "What ratio of kinetic energy is lost because of the collision?",
        "Approximately what fraction of kinetic energy vanishes in the collision?",
        "What is the proportion of kinetic energy lost in the collision?",
    ],
    "F_FORCES_COUNTING_PUSHED_OBJECTS": [
        "How many objects start moving because of a direct push?",
        "What number of objects are launched into motion by a direct push?",
        "Can you count the objects set moving via a direct push?",
        "How many items begin to move due to a direct push?",
        "How many objects are put into motion through a direct push?",
        "How many objects respond to a direct push by moving?",
        "How many bodies are driven into motion by a direct push?",
        "What is the count of objects that start moving from a direct push?",
        "How many objects get set moving by a direct push?",
        "How many objects are propelled into motion via a direct push?",
    ],
    "F_FORCES_COUNTING_BALANCED": [
        "How many objects experience balanced forces for the entire event?",
        "What number of objects stay under balanced forces during the event?",
        "Can you count the objects that maintain balanced forces throughout the event?",
        "How many items are subject to balanced forces the whole time?",
        "How many objects remain in equilibrium during the event?",
        "How many objects feel balanced forces for the duration of the event?",
        "What is the count of objects experiencing balanced forces throughout?",
        "How many bodies undergo balanced forces across the event?",
        "How many objects keep balanced forces acting on them the entire event?",
        "How many objects are under balanced force conditions throughout the event?",
    ],
    "F_FORCES_ATTRIBUTE_PRIMARY_FORCE": [
        "Which force mainly drives the object's motion?",
        "Identify the force chiefly responsible for the object's movement?",
        "Which force is the primary cause of the object's movement?",
        "Which force leads the object to move?",
        "Which force plays the dominant role in moving the object?",
        "Point out the force that primarily produces the object's motion?",
        "Which force is the main driver behind the object's movement?",
        "Name the force most responsible for the object's motion?",
        "Which force causes the object to move foremost?",
        "Which force is chiefly accountable for the object's movement?",
    ],
    "F_FORCES_ATTRIBUTE_DIRECTION_NET": [
        "Which direction does the net force push the target object?",
        "Identify the direction of the net force on the target object?",
        "In what direction is the net force applied to the target object?",
        "Which way does the net force act on the target object?",
        "Toward which direction does the net force drive the target object?",
        "Where is the net force pointing on the target object?",
        "Which direction does the target object's net force take?",
        "In which way does the net force influence the target object?",
        "Name the direction in which the net force acts on the target object?",
        "What direction does the net force have on the target object?",
    ],
    "F_FORCES_ATTRIBUTE_MAX_FRICTION": [
        "Which object is subject to the largest frictional force?",
        "Identify the object experiencing the most friction?",
        "Which item encounters the greatest frictional force?",
        "Which object feels the highest friction?",
        "Which object undergoes the maximum frictional force?",
        "Point out the object with the greatest friction acting on it?",
        "Which object is affected by the strongest frictional force?",
        "Name the object experiencing the highest friction?",
        "Which object deals with the greatest friction force?",
        "Which object is exposed to the maximum friction?",
    ],
    "F_FORCES_ESTIMATION_NET_FORCE": [
        "About how large is the net force on the target object?",
        "What is your estimate of the net force acting on the target object?",
        "Approximately what net force does the target object experience?",
        "Can you approximate the net force applied to the target object?",
        "What net force acts on the target object, roughly?",
        "Give an estimate of the target object's net force?",
        "What would you say is the net force on the target object?",
        "Estimate the magnitude of the net force acting on the target object?",
        "Approximately how much net force is applied to the target object?",
        "What is the rough net force affecting the target object?",
    ],
    "F_FORCES_ESTIMATION_TORQUE": [
        "About how much torque is applied around the object's pivot?",
        "What is your estimate of the torque acting about the object's pivot?",
        "Approximately what torque is applied about the object's pivot?",
        "Can you approximate the torque exerted around the object's pivot?",
        "What torque acts about the object's pivot, roughly?",
        "Give an estimate of the torque applied to the object's pivot?",
        "What would you say is the torque about the object's pivot?",
        "Estimate the magnitude of torque applied around the object's pivot?",
        "Approximately how much torque is placed on the object's pivot?",
        "What is the rough torque acting about the object's pivot?",
    ],
}

PLACEHOLDER_PATTERN = re.compile(r"<([A-Z0-9_]+)>")


def capitalize_first(text):
    return text[0].upper() + text[1:] if text else text


def ensure_question(text):
    text = text.strip()
    if not text.endswith('?'):
        text += '?'
    return capitalize_first(text)


def dedupe_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def add_relative_that(rest):
    lowered = rest.lower()
    if lowered.startswith('that '):
        return rest
    verbs = {
        'is', 'are', 'was', 'were', 'becomes', 'become', 'causes', 'makes',
        'leads', 'remains', 'stays', 'supports', 'forms', 'contains', 'lies',
        'sits', 'stands', 'ends', 'enters', 'exits', 'contacts', 'experiences',
        'appears', 'blocks', 'occludes', 'touches', 'hits', 'collides',
        'undergoes', 'shows', 'happens', 'occurs', 'lands', 'stops', 'results',
        'exists', 'falls', 'rotates', 'moves', 'travels', 'starts', 'begins'
    }
    first = rest.split(' ', 1)[0].lower() if rest else ''
    if first in verbs:
        return f"that {rest}"
    return rest


def generate_how_many(question):
    base = question.rstrip('?').strip()
    rest = base[len('How many '):].strip()
    rest_clause = add_relative_that(rest)
    rest_subject = rest_clause
    endings = [
        ' are there',
        ' are present',
        ' are visible',
        ' are currently visible',
        ' are in the scene',
        ' are within view',
    ]
    for ending in endings:
        if rest_clause.endswith(ending):
            rest_subject = rest_clause[:-len(ending)].strip()
            break
    templates = [
        f"How many {rest}?",
        f"Approximately how many {rest}?",
        f"About how many {rest}?",
        f"What is the count of {rest_subject}?",
        f"What number of {rest_subject} do you observe?",
        f"Could you tally the {rest_subject}?",
        f"Can you count the {rest_subject}?",
        f"Could you state how many {rest_subject} there are?",
        f"Would you report the number of {rest_subject}?",
        f"Could you give the total of {rest_subject}?",
        f"What total of {rest_subject} is present?",
        f"Roughly how many {rest_subject} are there?",
        f"Please indicate the total {rest_subject}.",
    ]
    return dedupe_preserve_order([ensure_question(t) for t in templates])[:10]


def generate_which_object(question):
    base = question.rstrip('?').strip()
    rest = base[len('Which object'):].strip()
    rest_clause = add_relative_that(rest)
    templates = [
        f"Which object {rest}?",
        f"Which item {rest}?",
        f"Which entity {rest}?",
        f"Which body {rest}?",
        f"Can you identify which object {rest}?",
        f"Could you specify the object that {rest_clause}?",
        f"Which object should be noted as the one that {rest_clause}?",
        f"Which object is regarded as the one that {rest_clause}?",
        f"Which object would you point out as the one that {rest_clause}?",
        f"Could you tell me which object {rest}?",
        f"Which object ends up being the one that {rest_clause}?",
        f"Which object ought to be labeled as the one that {rest_clause}?",
    ]
    return dedupe_preserve_order([ensure_question(t) for t in templates])[:10]


def generate_which_objects(question):
    base = question.rstrip('?').strip()
    rest = base[len('Which objects'):].strip()
    rest_clause = add_relative_that(rest)
    templates = [
        f"Which objects {rest}?",
        f"Which items {rest}?",
        f"Which entities {rest}?",
        f"Which bodies {rest}?",
        f"Can you identify which objects {rest}?",
        f"Could you specify the objects that {rest_clause}?",
        f"Which objects should be noted as those that {rest_clause}?",
        f"Which objects are regarded as the ones that {rest_clause}?",
        f"Which objects would you point out as the ones that {rest_clause}?",
        f"Could you tell me which objects {rest}?",
        f"Which objects end up being the ones that {rest_clause}?",
        f"Which objects ought to be labeled as those that {rest_clause}?",
    ]
    return dedupe_preserve_order([ensure_question(t) for t in templates])[:10]


def generate_which_general(question):
    base = question.rstrip('?').strip()
    rest = base[len('Which '):].strip()
    templates = [
        f"Which {rest}?",
        f"Which particular {rest}?",
        f"Which specific {rest}?",
        f"Which option for {rest} applies?",
        f"Can you indicate which {rest}?",
        f"Could you specify which {rest}?",
        f"Would you point out which {rest}?",
        f"Please identify which {rest}?",
        f"Which choice of {rest} is correct?",
        f"Which selection of {rest} fits best?",
        f"Could you tell me which {rest}?",
        f"Which version of {rest} should we consider?",
    ]
    return dedupe_preserve_order([ensure_question(t) for t in templates])[:10]


def generate_what_is_the(question):
    base = question.rstrip('?').strip()
    rest = base[len('What is the '):].strip()
    templates = [
        f"What is the {rest}?",
        f"How would you characterize the {rest}?",
        f"What best describes the {rest}?",
        f"Which option corresponds to the {rest}?",
        f"Could you identify the {rest}?",
        f"Can you specify the {rest}?",
        f"Which choice represents the {rest}?",
        f"What do you determine to be the {rest}?",
        f"Would you indicate the {rest}?",
        f"What answer points to the {rest}?",
        f"Please state the {rest}?",
        f"How might you describe the {rest}?",
    ]
    return dedupe_preserve_order([ensure_question(t) for t in templates])[:10]


def generate_what_is(question):
    base = question.rstrip('?').strip()
    rest = base[len('What is '):].strip()
    templates = [
        f"What is {rest}?",
        f"How would you describe {rest}?",
        f"What best characterizes {rest}?",
        f"Which option corresponds to {rest}?",
        f"Could you identify {rest}?",
        f"Can you specify {rest}?",
        f"Which choice represents {rest}?",
        f"What do you determine to be {rest}?",
        f"Would you indicate {rest}?",
        f"What answer points to {rest}?",
        f"Please state {rest}?",
        f"How might you explain {rest}?",
    ]
    return dedupe_preserve_order([ensure_question(t) for t in templates])[:10]


def generate_generic(question):
    question = question.strip()
    templates = [
        question,
        f"Please answer: {question}",
        f"I wonder, {question}",
        f"Could you clarify: {question}",
        f"Would you mind confirming: {question}",
        f"Kindly explain: {question}",
        f"Help me understand: {question}",
        f"Quick check\u2014{question}",
        f"Could you provide insight into this: {question}",
        f"Let me know: {question}",
        f"Clarify for me: {question}",
        f"Can you shed light on this question: {question}",
    ]
    return dedupe_preserve_order([ensure_question(t) for t in templates])[:10]


def generate_from_which(question):
    base = question.rstrip('?').strip()
    rest = base[len('From '):].strip()
    alt = f"Which {rest.replace(' does ', ' does ')}"
    templates = [
        base,
        f"From which {rest}?",
        f"Which {rest}?",
        f"Could you specify from which {rest}?",
        f"Can you tell me from which {rest}?",
        f"Please indicate from which {rest}?",
        f"Would you note from which {rest}?",
        f"From what {rest} does this occur?",
        f"Identify from which {rest}?",
        f"Explain from which {rest}?",
        f"Share from which {rest}?",
        f"Clarify from which {rest}?",
    ]
    return dedupe_preserve_order([ensure_question(t) for t in templates])[:10]


def generate_where(question):
    base = question.rstrip('?').strip()
    rest = base[len('Where '):].strip()
    templates = [
        f"Where {rest}?",
        f"Precisely where {rest}?",
        f"Can you indicate where {rest}?",
        f"Could you specify where {rest}?",
        f"Please point out where {rest}?",
        f"Where exactly {rest}?",
        f"Would you note where {rest}?",
        f"Identify where {rest}?",
        f"Clarify where {rest}?",
        f"In what place {rest}?",
        f"At which location {rest}?",
        f"Where would you say {rest}?",
    ]
    return dedupe_preserve_order([ensure_question(t) for t in templates])[:10]


def generate_when(question):
    base = question.rstrip('?').strip()
    rest = base[len('When '):].strip()
    templates = [
        f"When {rest}?",
        f"Exactly when {rest}?",
        f"At what time {rest}?",
        f"Could you specify when {rest}?",
        f"Can you tell me when {rest}?",
        f"Please indicate when {rest}?",
        f"Would you note when {rest}?",
        f"Clarify when {rest}?",
        f"Identify when {rest}?",
        f"When precisely {rest}?",
        f"When does it happen that {rest}?",
        f"When would you say {rest}?",
    ]
    return dedupe_preserve_order([ensure_question(t) for t in templates])[:10]


def generate_for_how_long(question):
    base = question.rstrip('?').strip()
    rest = base[len('For how long '):].strip()
    templates = [
        f"For how long {rest}?",
        f"About how long {rest}?",
        f"Approximately how long {rest}?",
        f"Can you say how long {rest}?",
        f"Could you estimate how long {rest}?",
        f"Please indicate how long {rest}?",
        f"Would you note how long {rest}?",
        f"Tell me how long {rest}?",
        f"Clarify how long {rest}?",
        f"What duration describes how long {rest}?",
        f"How much time passes while {rest}?",
        f"Give an idea of how long {rest}?",
    ]
    return dedupe_preserve_order([ensure_question(t) for t in templates])[:10]


def generate_during_which(question):
    base = question.rstrip('?').strip()
    rest = base[len('During '):].strip()
    templates = [
        base,
        f"During which {rest}?",
        f"In which {rest}?",
        f"Across which {rest}?",
        f"Through which {rest}?",
        f"Could you specify during which {rest}?",
        f"Can you indicate during which {rest}?",
        f"Please identify during which {rest}?",
        f"Would you note during which {rest}?",
        f"Clarify during which {rest}?",
        f"During what {rest} does this occur?",
        f"Which {rest} does this take place during?",
    ]
    return dedupe_preserve_order([ensure_question(t) for t in templates])[:10]


def auto_generate_variations(question):
    question = question.strip()
    lower = question.lower()
    if lower.startswith('how many '):
        variations = generate_how_many(question)
    elif lower.startswith('which object '):
        variations = generate_which_object(question)
    elif lower.startswith('which objects '):
        variations = generate_which_objects(question)
    elif lower.startswith('which '):
        variations = generate_which_general(question)
    elif lower.startswith('what is the '):
        variations = generate_what_is_the(question)
    elif lower.startswith('what is '):
        variations = generate_what_is(question)
    elif lower.startswith('where '):
        variations = generate_where(question)
    elif lower.startswith('when '):
        variations = generate_when(question)
    elif lower.startswith('for how long '):
        variations = generate_for_how_long(question)
    elif lower.startswith('from '):
        variations = generate_from_which(question)
    elif lower.startswith('during '):
        variations = generate_during_which(question)
    else:
        variations = generate_generic(question)

    if len(variations) < 10:
        supplemental = generate_generic(question)
        for candidate in supplemental:
            if candidate not in variations:
                variations.append(candidate)
            if len(variations) == 10:
                break
    return variations[:10]


def fill_template(text, attributes):
    if not text or not attributes:
        return text

    def _replace(match):
        key = match.group(1)
        value = attributes.get(key)
        return str(value) if value is not None else match.group(0)

    return PLACEHOLDER_PATTERN.sub(_replace, text)


# Normalize phrasing
def normalize_question(text):
    replacements = [
        ("Identify the ", "Could you identify the "),
        ("Identify ", "Could you identify "),
        ("Point out the ", "Could you point out the "),
        ("Point out ", "Could you point out "),
        ("Point to the ", "Could you point to the "),
        ("Point to ", "Could you point to "),
        ("Name the ", "Could you name the "),
        ("Name ", "Could you name "),
        ("Describe ", "How would you describe "),
        ("Give an estimate of", "Could you give an estimate of"),
        ("Give an estimate for", "Could you give an estimate for"),
        ("Give an estimate", "Could you give an estimate"),
        ("Give an ", "Could you give an "),
        ("Estimate the ", "What is the estimated "),
        ("Estimate how ", "How would you estimate how "),
        ("Estimate ", "Could you estimate "),
    ]
    for old, new in replacements:
        if text.startswith(old):
            text = new + text[len(old):]
            break
    return text

for key, items in variations.items():
    variations[key] = [normalize_question(q) for q in items]


# Validate uniform list lengths
for key, items in variations.items():
    if len(items) != 10:
        raise ValueError(f"{key} has {len(items)} variations")

new_data = {}
for category, entries in data.items():
    new_category = {}
    for qid, info in entries.items():
        base = {k: v for k, v in info.items()}
        attributes = info.get('attributes', {})
        qs = variations.get(qid)
        if qs is None:
            qs = [normalize_question(q) for q in auto_generate_variations(info['question'])]
        base_question_already_filled = False
        for idx, question in enumerate(qs, 1):
            new_qid = f"{qid}_{idx}"
            entry = base.copy()
            entry['question'] = fill_template(question, attributes)
            new_category[new_qid] = entry
    new_data[category] = new_category

with open('simpler_extended_variation.json', 'w') as f:
    json.dump(new_data, f, indent=4, ensure_ascii=True)
