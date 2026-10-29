"""Words for the stage 6-8 report pages (scripts/38_wm3_reports.py). Numbers are from
outputs/wm3/STEPS.md and the stage JSON files."""

import json

from world import cities

W3 = cities.OUTPUTS / "wm3"

SHOCK_LABEL = {"supply60": "Shock: 60% of idle drivers log off", "supply30": "Shock: 30% of idle drivers log off",
               "price15": "Shock: prices x1.5 everywhere"}
SHOCK_SUB = {"supply60": "at 12:00, 60% of the idle drivers log off for good", "supply30": "at 12:00, 30% of the idle drivers log off for good",
             "price15": "from 12:00 every price is at least 1.5 times normal"}

SHOCK_LOOK = {
    "supply60": "<br><b>What to look for:</b> drivers online fall by the same amount on both maps, and the extra waiting riders and the lost pickups build up together (charts on the right).",
    "supply30": "<br><b>What to look for:</b> a milder version of the 60% shock; both maps lose drivers and pickups at about the same pace.",
    "price15": "<br><b>What to look for:</b> the model's known weak spot. In the simulator many more riders see the 1.5x price and decline it "
               "(red circles; chart 'riders who declined the price') than the model imagines, so the model keeps too many pickups.",
}

REPLAY_NOTE = {
    6: "The same day, with the same riders and drivers, is run twice in the simulator. Until 12:00 both runs are identical. "
       "From 12:00 the <b>left map</b> is run by the platform's hand-written rule and the <b>right map</b> by the world-model agent, "
       "which decides which driver goes to which rider (every 30 s) and which idle drivers should move to another area (every 5 min). "
       "Everything on both maps is what the simulator really did with those decisions; nothing here is imagined.",
    7: "The same day is run twice in the simulator, identical until 12:00; both use the same platform (the stage-6 agent). "
       "<b>Left:</b> each free driver gets one ride offer at a time, and the simulated people (behaviour learned from data) accept or reject it. "
       "<b>Right:</b> every free driver sees a <b>menu</b> of up to 4 rides with the fare it would earn, and the trained driver agents choose. "
       "The panel on the right shows one driver's menu at a time; click any driver to see its own. "
       "When few riders are waiting, many idle drivers see the same ride and all pick it: the platform gives it to the closest one "
       "(bold line) and the others lose it (faint dotted lines) and get a new menu 30 s later. That is why bursts of lines meet at one rider.",
    8: "<b>Left:</b> the simulator runs the day with the shock. <b>Right:</b> the world model is given only the city's state at 12:00 and the shock, "
       "then imagines the next hour on its own (the riders, the drivers' choices, the cancellations are all its own predictions). "
       "Individual drivers will not match one by one - the model draws one possible future - but the city's response should: "
       "compare the counters under the maps and the charts on the side. Use the menu above to switch shocks.",
}

LEGEND_BASE = ('<span><span class="sw" style="background:var(--idle)"></span>idle driver</span>'
               '<span><span class="sw" style="background:var(--blue)"></span>driver offered a ride / driving to the rider</span>'
               '<span><span class="sw" style="background:var(--orange)"></span>driver with a rider</span>'
               '<span><span class="sw" style="background:var(--aqua)"></span>driver moving to another area</span>'
               '<span><span class="sw" style="border:1.5px solid var(--idle)"></span>on a break</span>'
               '<span><b>&times;</b> rider waiting for a driver</span>'
               '<span><span class="sw" style="border:1.5px solid var(--red)"></span>rider saw the price and declined</span>')
LEG_OFFER = ('<span><span class="ln" style="border-color:var(--blue)"></span>ride offer, accepted (driver &rarr; rider)</span>'
             '<span><span class="ln" style="border-color:var(--red);border-top-style:dashed"></span>offer rejected by the driver</span>'
             '<span><span class="ln" style="border-color:var(--idle);border-top-style:dashed"></span>offer not answered</span>')
LEG_MOVE = ('<span><span class="ln" style="border-color:var(--aqua)"></span>&#9656; order to move to an area, followed</span>'
            '<span><span class="ln" style="border-color:var(--idle);border-top-style:dashed"></span>&#9656; move order declined</span>')
LEG_MENU = ('<span><span class="ln" style="border-color:var(--ink3)"></span>a ride on a driver\'s menu (right map)</span>'
            '<span><span class="ln" style="border-color:var(--blue);border-width:3px"></span>the ride the driver took</span>'
            '<span><span class="ln" style="border-color:var(--blue);border-top-style:dotted"></span>took it, but a closer driver won it</span>'
            '<span><span class="sw" style="border:2px solid var(--red)"></span>declined the whole menu</span>')
LEG_SHOCK = '<span><b style="color:var(--red)">&times;</b> driver removed by the shock (first 10 min)</span>'

HOWTO = ("<li><b>Play</b> runs the hour; the slider jumps to any minute. Scroll to zoom, drag to pan; both maps move together.</li>"
         "<li><b>Click a driver</b> on either map to follow it: the panel lists everything that happened to it. <b>auto</b> follows drivers who are acting right now.</li>"
         "<li><b>Actions, last 2 minutes</b> lists the decisions just taken on the left or right map; click a line to follow that driver.</li>"
         "<li>The small charts compare the two maps since 12:00 (faint line = the full hour, bold = up to now).</li>"
         "<li>The buttons above the maps hide or show each kind of mark.</li>")

PAGE = {
    6: {"title": "Stage 6 - Matching",
        "replay": {"layers": [["offers", "ride offers"], ["moves", "move orders"], ["riders", "waiting riders"], ["events", "declined prices"]],
                   "legend": LEGEND_BASE + LEG_OFFER + LEG_MOVE, "howto": HOWTO,
                   "mini": [{"key": "pickups", "label": "riders picked up"}, {"key": "waiting", "label": "riders waiting now"},
                            {"key": "rejected", "label": "offers drivers rejected or ignored"}, {"key": "moves", "label": "move orders sent"}]}},
    7: {"title": "Stage 7 - Driver agents",
        "replay": {"layers": [["menus", "menus"], ["offers", "single offers"], ["moves", "move orders"], ["riders", "waiting riders"], ["events", "declined prices"]],
                   "legend": LEGEND_BASE + LEG_OFFER + LEG_MENU + LEG_MOVE, "howto": HOWTO,
                   "mini": [{"key": "pickups", "label": "riders picked up"}, {"key": "waiting", "label": "riders waiting now"},
                            {"key": "rejected", "label": "single offers rejected or ignored"}, {"key": "cancels", "label": "riders who gave up"}]}},
    8: {"title": "Stage 8 - World-model test",
        "replay": {"layers": [["events", "shock marks"], ["offers", "ride offers"], ["moves", "move orders"], ["riders", "waiting riders"]],
                   "legend": LEGEND_BASE + LEG_SHOCK + LEG_OFFER + LEG_MOVE, "howto": HOWTO,
                   "mini": [{"key": "pickups", "label": "riders picked up"}, {"key": "waiting", "label": "riders waiting now"},
                            {"key": "declined", "label": "riders who declined the price"}, {"key": "online", "label": "drivers online"}]}},
}


def fig(src, title, shows, take):
    sh = "".join(f"<li>{x}</li>" for x in shows)
    tk = "".join(f"<li>{x}</li>" for x in take)
    return (f'<div class="fig"><h3 style="margin-top:0">{title}</h3><img src="{src}" alt="{title}" loading="lazy">'
            f'<div class="cap"><div><h4>What it shows</h4><ul>{sh}</ul></div><div><h4>What to take from it</h4><ul>{tk}</ul></div></div></div>')


def flow(steps):
    out = []
    for i, (b, t) in enumerate(steps):
        if i:
            out.append('<span class="arr">&rarr;</span>')
        out.append(f'<div class="box"><b>{b}</b>{t}</div>')
    return '<div class="flow">' + "".join(out) + "</div>"


def table(head, rows):
    h = "".join(f"<th>{x}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f'<table class="t"><tr>{h}</tr>{b}</table>'


GLOSS = ('<p class="muted" style="font-size:14px"><b>Words used:</b> <i>reward</i> = riders picked up in the 2 hours + drivers\' welfare '
         '(a fairness-weighted sum of each driver\'s net earnings); <i>&plusmn;</i> = standard error over the test days (seeds); '
         '<i>pretrained</i> = the decision maker is built on the trained world model; <i>scratch</i> = the same network from random weights; '
         '<i>imagination</i> = the world model predicting the city step by step, without the simulator.</p>')


def content(stage, png):
    return {6: s6, 7: s7, 8: s8}[stage](png)


# ------------------------------------------------------------------------------------------- stage 6
def s6(png):
    d = W3 / "stage6"
    intro = f"""
<h1>Stage 6: the world model decides which driver gets which rider</h1>
<p class="sub">World model 3 · ride_world · San Francisco and Pune</p>
<p>Every 30 seconds a ride platform must decide which free driver goes to which waiting rider (<b>matching</b>), and every 5 minutes
which idle drivers it should send to areas where riders are expected (<b>repositioning</b>). Normally a hand-written rule does both:
the closest driver gets the rider, and drivers are nudged towards busy areas. In stage 5 the world model learned to do the repositioning.
In stage 6 it also does the matching, and its decisions are carried out in the real simulator.</p>
<div class="short"><b>In short</b><ul>
<li><span class="good">It matches as well as the platform's rule.</span> With the same repositioning, its own matching changed the reward by
 &minus;1.1 &plusmn; 3.9 in San Francisco and +0.3 &plusmn; 5.5 in Pune: no difference beyond noise.</li>
<li>Drivers rejected <b>fewer</b> of its offers: about 11 fewer per 2 hours in San Francisco and 19 fewer in Pune.</li>
<li>It learned by copying the rule inside the world model's imagination (it picks exactly the rule's driver 86-90% of the time).
 Extra reinforcement learning on matching did not improve it further.</li>
<li><span class="good">Building on the pretrained world model pays:</span> the same network trained from scratch earns 11 reward less in San Francisco (+11.3 &plusmn; 3.5, better on 15 of 20 days).</li>
<li>In Pune nothing beats "do nothing" clearly; the rule's small edge there comes from its repositioning, not its matching.</li>
</ul></div>
<h3>How the agent was made</h3>
{flow([("1. Imagine", "the world model replays the city from 144 real starting moments per city"),
       ("2. Copy the rule", "in imagination, every waiting rider learns to pick the same driver the rule would pick (15k choices SF, 26k Pune)"),
       ("3. Try to improve", "reinforcement learning (GRPO): try other drivers for one rider, keep what earns more reward 20 min later"),
       ("4. Run for real", "the agent decides inside the simulator on unseen days, 12:00-14:00 (20 SF + 12 Pune days)"),
       ("5. Compare", "same days with the platform's rule, and with no repositioning")])}
"""
    body = f"""
<h2>Results</h2>
<p>Each number is the 2-hour effect compared with the same day run with the rule's matching and <i>no</i> repositioning (mean &plusmn; standard error over days).</p>
{table(["who decides", "SF reward", "SF pickups", "Pune reward", "Pune pickups"],
       [["platform rule (matching + repositioning)", "+25.0 &plusmn; 6.8", "+25.4", "+7.7 &plusmn; 7.0", "+11.7"],
        ["stage 5: rule matches, agent moves drivers", "+23.8 &plusmn; 5.7", "+23.3", "&minus;1.0 &plusmn; 6.2", "&minus;1.0"],
        ["<b>stage 6: agent matches and moves, pretrained</b>", "<b>+22.7 &plusmn; 6.0</b>", "<b>+22.4</b>", "<b>&minus;0.7 &plusmn; 6.0</b>", "<b>&minus;4.1</b>"],
        ["stage 6: agent matches and moves, from scratch", "+11.5 &plusmn; 4.5", "+11.4", "+3.1 &plusmn; 6.2", "+3.8"]])}
{GLOSS}
<h2>The figures, explained</h2>
{fig(png(d / "34_execute.png"), "Carried out in the simulator: every decision maker on unseen days",
     ["Top row San Francisco, bottom row Pune. Each panel is one measure over the 2-hour window, as a difference from the rule's matching with no repositioning.",
      "Bars = mean over days, with standard error; black dots = individual days; open diamonds = what the world model <i>imagined</i> for the same decision maker.",
      "rule = platform's rule; stage 5 = rule matches, agent repositions; IM-p = agent does both, pretrained; IM-s = agent does both, from scratch."],
     ["San Francisco: the pretrained agent (dark orange) reaches the rule's reward and pickups; the scratch agent (green) gets about half.",
      "Offers rejected: the pretrained agent's offers are turned down less often than the rule's.",
      "Pickup minutes of offers and rider waits barely move: matching closest-first is already near the best; the agent learned to do the same.",
      "Pune: all bars sit within the scatter of the dots: no decision maker clearly helps there in 2 hours.",
      "Diamonds close to bars mean the world model predicted these effects well before anything was run."])}
{fig(png(d / "34_imitation.png"), "Step 2: copying the rule's matching in imagination",
     ["Training curves over 6 passes through the data; dark orange = built on the pretrained world model, green = from scratch.",
      "1st column: how surprised the agent is by the rule's choice (lower is better). 2nd: how often it picks exactly the rule's driver.",
      "3rd: pickup minutes of the driver it picks vs the rule's (dashed). 4th: how far its repositioning drifted from stage 5 (should stay near 0)."],
     ["After one pass the agent already picks the rule's driver ~85% (SF) / ~89% (Pune) of the time, and when it differs it picks a driver just as close (2.0 vs 1.96 min SF).",
      "Its repositioning is unchanged (drift below 0.02), so stage 6 adds matching without losing stage 5."])}
{fig(png(d / "34_training.png"), "Step 3: trying to improve matching with reinforcement learning (in imagination)",
     ["40 iterations of GRPO; each point is the imagined reward over 60 minutes from held-out starting moments, compared with rule matching and no repositioning.",
      "Solid = the stage-6 agent; dotted = stage-5 agent (rule matches); dashed blue = the platform's rule. Last column: how often the agent chooses to hold a rider back."],
     ["The solid line does not climb: SF +4.2 &rarr; +4.3, Pune +0.4 &rarr; &minus;2.3. Reinforcement learning found nothing better than closest-first.",
      "Why: once matching is closest-first, which of two nearby drivers gets one rider changes little an hour later; the gains are in where idle drivers wait (stage 5).",
      "The imitation agent (before RL) is therefore the one carried out in the simulator above."])}
"""
    return intro, body


# ------------------------------------------------------------------------------------------- stage 7
def s7(png):
    d = W3 / "stage7"
    intro = f"""
<h1>Stage 7: drivers become agents that choose their own rides</h1>
<p class="sub">World model 3 · ride_world · San Francisco and Pune</p>
<p>Until now the platform sent each driver one ride at a time, and simulated people (whose behaviour was learned from data) accepted or rejected it.
In stage 7 the platform instead shows every free driver a <b>menu</b>: up to 4 rides, each with the pickup time, the trip length and exactly what the
driver would earn. Every driver is now an <b>agent with its own goal: earn as much as possible over its shift</b>. One small policy (about 10,000 weights),
shared by all drivers, was trained with reinforcement learning inside the world model's imagination; each driver was rewarded only by its own earnings.</p>
<div class="short"><b>In short</b><ul>
<li><b>Drivers learned to never decline and to take the platform's first (closest) ride</b>: 98% of the time in San Francisco, 100% in Pune.
 Driving unpaid to a pickup is what costs a driver most, so self-interest agrees with the platform's ordering.</li>
<li><span class="good">Menus with these drivers shorten rider waits</span> (SF &minus;0.25 min, Pune &minus;0.15 min) and add pickups in Pune
 (+22 per 2 hours) compared with single offers to simulated people, who reject 22% of offers in Pune.</li>
<li>Drivers' earnings per online hour stay the same in San Francisco and rise 1% in Pune: rides are limited by riders, not drivers.</li>
<li><span class="warn">Selfish drivers cost the city a little in Pune:</span> trained drivers vs drivers who always take option 1:
 &minus;10 &plusmn; 5 pickups per 2 hours. The world model predicted this in imagination before it was run.</li>
</ul></div>
<h3>How a menu round works (every 30 seconds)</h3>
{flow([("1. Platform scores", "the stage-6 agent scores every (waiting rider, free driver) pair"),
       ("2. Menus", "each free driver sees its 4 best rides: pickup min, trip min, fare after commission, net per hour"),
       ("3. Drivers choose", "the driver policy picks a ride or declines; it sees the options and its own day (hours online, earnings so far)"),
       ("4. Conflicts", "a ride chosen by several drivers goes to the one the platform rates best; the others lose it"),
       ("5. Simulator", "the taken rides are driven; riders, traffic and everyone else continue as before")])}
"""
    body = f"""
<h2>Results</h2>
<p>Difference from <b>single offers answered by the simulated people</b> (same platform), over 2-hour windows on unseen days (mean &plusmn; standard error).</p>
{table(["drivers", "SF pickups", "SF rider wait", "Pune pickups", "Pune rider wait", "Pune net per hour"],
       [["menus, drivers always take option 1 (start)", "&minus;1.0 &plusmn; 4.2", "&minus;0.18 &plusmn; .03 min", "+31.8 &plusmn; 4.3", "&minus;0.23 &plusmn; .03 min", "+0.6%"],
        ["<b>menus, trained earning-maximising drivers</b>", "<b>&minus;1.5 &plusmn; 3.1</b>", "<b>&minus;0.25 &plusmn; .04 min</b>", "<b>+21.8 &plusmn; 6.6</b>", "<b>&minus;0.15 &plusmn; .02 min</b>", "<b>+0.5%</b>"]])}
<p>Trained vs start drivers, same days: own earnings per hour equal in both cities; the poorest 10% of drivers slightly up; in Pune riders lose
(&minus;10.1 &plusmn; 5.1 pickups, +0.08 min wait). Each driver gains a little by never passing a ride, and the city does slightly worse.</p>
{GLOSS}
<h2>The figures, explained</h2>
{fig(png(d / "34_execute.png"), "Carried out in the simulator: menus vs single offers",
     ["Each panel: a difference from single offers answered by the simulated people (same platform, same days), over 2 hours. Top San Francisco, bottom Pune.",
      "Grey = menus where every driver takes the first option (the start policy); orange = menus with the trained earning-maximising drivers. Dots = days.",
      "Measures: drivers' net per online hour, the 10th-percentile driver's net per hour, riders picked up, rider wait, drivers' welfare, inequality of income (Gini)."],
     ["Rider wait drops in both cities with either kind of menu driver: nobody rejects a ride and has to be re-offered.",
      "San Francisco pickups and earnings: unchanged. Pune pickups: up 32 (start) and 22 (trained) per 2 hours.",
      "Trained vs start in Pune: fewer pickups (orange bar lower than grey) at no gain to the drivers' average: a small cost of selfish choices."])}
{fig(png(d / "34_drivers_training.png"), "Training the drivers in imagination (GRPO)",
     ["40 iterations; each point is 60 imagined minutes from held-out starting moments.",
      "Orange = trained drivers; grey dashed = drivers who take option 1; blue dashed = single offers with simulated people.",
      "Columns: drivers' net per online hour, 10th-percentile driver, riders picked up, drivers' welfare."],
     ["San Francisco: the lines stay within a narrow band: there is little to gain for a driver beyond taking the closest ride.",
      "Pune: as the drivers learn, riders picked up and welfare fall (about &minus;6 pickups per hour). The world model warned of this in imagination,",
      "and the simulator confirmed it (&minus;10 &plusmn; 5 per 2 hours): the world model can be used to test a policy's side effects before running it."])}
"""
    return intro, body


# ------------------------------------------------------------------------------------------- stage 8
def scorecard_table():
    sc = json.loads((W3 / "stage8" / "37_scorecard.json").read_text())["criteria"]
    cls = {"met": "good", "partly": "warn", "not met": "bad"}
    ic = {"met": "&#9679; met", "partly": "&#9679; partly", "not met": "&#9679; not met"}
    rows = [[f"<b>{c['criterion']}</b><br><span class='muted'>{c['question']}</span>", f"<span class='{cls[c['verdict']]}'>{ic[c['verdict']]}</span>",
             c["sf_hd"], c["pune_hd"], f"<span class='muted'>{c['note']}</span>"] for c in sc]
    t = table(["criterion", "verdict", "San Francisco", "Pune", "note"], rows)
    return t.replace('<td>', '<td style="text-align:left">').replace('<th>', '<th style="text-align:left">')


def s8(png):
    d = W3 / "stage8"
    intro = f"""
<h1>Stage 8: is wm3 a world model?</h1>
<p class="sub">World model 3 · ride_world · San Francisco and Pune</p>
<p>A <b>world model</b>, in the sense of Ha &amp; Schmidhuber's <i>World Models</i>, Dreamer, MuZero and LeCun's proposals, is a learned simulator:
from the current state it predicts how the world unfolds, responds to actions, answers what-if questions, and lets an agent learn or plan inside it.
We tested wm3 against ten criteria, each on days and seasons it never saw.</p>
<div class="short"><b>In short: close. wm3 is a <i>hybrid</i> world model</b>: every person's decisions and all demand are learned; the road network, fares, surge and request rules are coded.
<ul>
<li><span class="good">Met (5):</span> it represents the whole state; predicts the next 30 s well; responds to the platform's orders as the simulator does;
 policies trained inside it work in the simulator, and it ranks them correctly (26 of 27 clear pairs in San Francisco); its knowledge transfers to new tasks.</li>
<li><span class="warn">Partly (4):</span> 8-hour imagination stays on track but drifts high (+6-12% after ~3 h); it predicts supply shocks at the right size
 but price shocks at only &frac14;-&frac13; of the size; new days and seasons work, a new city is untested; the mechanics are coded, not learned.</li>
<li><span class="bad">Not met (1):</span> its imagined futures are too alike: the real day falls inside their range 51% (SF) / 33% (Pune) of the time instead of 78%.</li>
</ul></div>
<h2>Scorecard</h2>
{scorecard_table()}
"""
    body = f"""
<h2>8a. Imagining 8 hours ahead</h2>
<p>From the real state at 06:30 on unseen days (13 in San Francisco, 8 in Pune), the world model imagines until 14:30 (960 steps of 30 s) eight times,
with the platform's usual rule in charge and no knowledge of the real future. Every 5 minutes the imagined city is compared with the real one, and with two simple guesses:
<b>persistence</b> (things stay as at 06:30) and the <b>historical average</b> (the same clock time on other days).</p>
{table(["relative error at 1 h / 4 h / 8 h", "world model", "persistence", "historical average"],
       [["SF riders picked up", ".09 / .08 / .10", ".29 / .50 / .50", ".25 / .19 / .12"],
        ["SF drivers online", ".04 / .07 / .11", ".19 / .42 / .27", ".12 / .12 / .13"],
        ["SF cancellations", ".30 / .71 / .53", ".36 / .56 / .48", ".41 / .35 / .21"],
        ["Pune riders picked up", ".04 / .06 / .07", ".18 / .34 / .34", ".09 / .09 / .05"],
        ["Pune drivers online", ".06 / .05 / .04", ".31 / .51 / .51", ".14 / .13 / .12"],
        ["Pune cancellations", ".16 / .28 / .30", ".23 / .52 / .47", ".26 / .28 / .17"]])}
{fig(png(d / "35_long_fan.png"), "One day per city: the real day and eight imagined ones",
     ["Black = what the simulator did; thin blue = the world model's 8 imagined days; orange dashed = historical average.",
      "Columns: drivers online, idle, with a rider, riders waiting, then running totals of new requests, pickups, completed trips, cancellations, log-ons, log-offs."],
     ["The imagined days follow the real day's shape for 8 hours: the morning rise, the midday peak, the afternoon dip.",
      "Running totals (requests, pickups, cancellations) end above the black line: the model slowly over-predicts after ~3 h.",
      "The blue lines bunch together: the model is too sure of itself (see the rank chart below)."])}
{fig(png(d / "35_long_errors.png"), "Error over 8 hours, against two simple guesses",
     ["Each panel: average relative error over all test days, as imagination goes on. Blue = world model; grey = persistence; orange = historical average.",
      "Lower is better."],
     ["The world model beats persistence almost everywhere and for the whole 8 hours: it never blows up from feeding on its own predictions.",
      "It beats the historical average strongly in the first hours (it knows today's state), and they meet after ~6-8 h, when today's state stops mattering.",
      "Cancellations are its weakest quantity (up to +50% too many in SF after 3 h)."])}
{fig(png(d / "35_long_rank.png"), "Is the spread of the imagined futures right?",
     ["For every quantity and moment, where does the real value fall among the 8 imagined values (rank 0 = below all, 8 = above all)?",
      "If the spread were right, every rank would be equally common (dashed line)."],
     ["Both cities show a U: the real value is often outside all 8 imagined values. The imagined futures are too alike.",
      "San Francisco leans to rank 0 (the model is too high); Pune has both ends (too narrow both ways).",
      "Cause: the model has no notion of &ldquo;today is a busy/quiet day&rdquo;; all 8 futures share one demand level. A day-level factor inferred from the first hour would fix this."])}

<h2>8b. What-if: shocks it never saw</h2>
<p>At 12:00 on 20 SF and 8 Pune unseen days, three shocks no training day had: 30% or 60% of idle drivers log off for good (an outage, a strike),
or every price is held at 1.5 times normal for 2 hours. The simulator runs each shock; the world model imagines it from the 12:00 state.
Both are compared with the same day without the shock (the replay above shows one of these days).</p>
{table(["effect over 2 h (shock minus no shock)", "SF real", "SF imagined", "Pune real", "Pune imagined"],
       [["30% of drivers off: pickups", "&minus;40", "&minus;33", "&minus;78", "&minus;78"],
        ["60% of drivers off: pickups", "&minus;102", "&minus;80", "&minus;203", "&minus;219"],
        ["60% of drivers off: rider wait", "+1.24 min", "+1.22 min", "+1.33 min", "+1.35 min"],
        ["60% of drivers off: net per online hour", "+0.52", "+0.52", "+0.24", "+0.23"],
        ["prices x1.5: pickups", "&minus;84", "<span class='bad'>&minus;28</span>", "&minus;492", "<span class='bad'>&minus;124</span>"],
        ["prices x1.5: riders declining the price", "+113", "<span class='bad'>+50</span>", "+666", "<span class='bad'>+259</span>"]])}
{fig(png(d / "36_shock_effects.png"), "Effect of each shock: simulator vs world model",
     ["Each group of two bars is one shock: dark = what the simulator did, blue = what the world model imagined (mean &plusmn; s.e. over days).",
      "Columns: pickups, riders who gave up, riders who declined the price, rider wait, riders waiting, drivers online, drivers' net per online hour."],
     ["Driver shocks: the pairs of bars match almost everywhere, including side effects (more cancellations, longer waits, higher earnings for the drivers who stay).",
      "Price shock: the direction is right for every measure, but the size is far too small: riders decline a 1.5x price much more than the model learned.",
      "Why: in the data prices were mostly 1.0-1.3x; the model has to extrapolate. Days with randomised prices in the training data would fix it."])}
{fig(png(d / "36_shock_traj.png"), "How the effect builds up after the shock",
     ["Minutes after the shock on the x-axis; effect on riders waiting, running total of pickups, and running total of declined prices.",
      "Dark = simulator, blue = world model; solid = 30% drivers off, dashed = 60% off, dotted = prices x1.5."],
     ["For the driver shocks the blue and dark lines grow together minute by minute, not just at the end.",
      "For the price shock the gap opens from the first minutes: the error is in how riders react to the price, not a slow drift."])}
{fig(png(d / "36_shock_scatter.png"), "Every day and shock: predicted vs true effect",
     ["One point per day and shock; x = simulator's effect, y = world model's; dashed = perfect prediction."],
     ["Cancellations (r .91), declined prices (r .94) and waits (r .92): the model tells big-effect days from small-effect days.",
      "Pickups (r .59): driver shocks lie on the line; the Pune price-shock triangles lie far above it (predicted loss too small)."])}

<h2>8c. Decisions made in imagination, checked in the world</h2>
{fig(png(d / "37_ranking.png"), "Every decision maker from stages 5-7: imagined effect vs real effect",
     ["One point per decision maker (stages 5, 6 and 7); x = its effect in the simulator, y = what the world model imagined for it (both vs no repositioning, 2 h), with standard errors.",
      "Dashed = perfect agreement. 'Clear pairs' are pairs of decision makers whose real difference is clearly larger than noise."],
     ["San Francisco: rank correlation .72 over 11 decision makers, and 26 of the 27 clear pairs are in the same order in imagination. A planner could pick among them without running the simulator.",
      "Imagination flatters the menu policies (+33 imagined vs +20 real) and the copied agents: the world model is kinder to decision makers that differ from what it saw in training.",
      "Pune: real effects are mostly noise; the 2 clear pairs are ordered correctly."])}

<h2>What would close the gap</h2>
<ol>
<li><b>A day-level factor</b> inferred from the first hour ("today is busy"), so imagined futures spread like real days and drift less.</li>
<li><b>Stop the demand part from feeding on its own imagined requests</b>, the likely cause of the slow upward drift (not yet tested).</li>
<li><b>Training days with randomised prices</b>, so it learns how riders react to prices it rarely saw.</li>
<li>Then learn travel times instead of using the road graph's rules, and test on a new city.</li>
</ol>
"""
    return intro, body
