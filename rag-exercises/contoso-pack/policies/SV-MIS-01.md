---
id: SV-MIS-01
title: Missed deliveries
owner: Lee Wong, Delivery Operations Manager
revised: 2026-03-16
access: team
class: Internal
country: All
---

# SV-MIS-01 Missed deliveries

## 1 Purpose
This document tells Contoso Water drivers and the service desk what to do when a delivery cannot be made. It covers the steps for the driver, the reason codes, the card left for the client, rebooking, and when the account manager calls the client.

It applies to all drivers and service desk staff in London, Utrecht and Cologne. It is an internal document and is not shared with clients.

The owner of this document is Lee Wong, Delivery Operations Manager, London.

## 2 What counts as a missed delivery
A delivery is missed when the driver arrives at the client's site but cannot leave the bottles or finish the visit.

Common examples:

- The loading bay is closed and there is no other way in.
- The door code does not work and nobody answers.
- There is nobody at the office to sign for the delivery.
- The address on the route is wrong.

A delivery is also missed when the driver never reaches the site, for example because the van breaks down.

A delivery that is only late, but still done the same day between 07:00 and 16:00, is not a missed delivery.

## 3 Driver steps at the site
When a driver cannot make a delivery, the driver follows these steps in order:

1. Try every entrance named in the client's access instructions.
2. Call the site contact listed on the route. Wait up to 10 minutes for an answer.
3. If there is still no way in, call the service desk so they can try the client too.
4. If nobody can help within 15 minutes of arrival, record the delivery as missed.
5. Log result=MISSED on the delivery app with one reason code.
6. Leave a missed delivery card.

Drivers must never leave bottles outside or in a public area without a signature. Drivers also never carry more than one bottle by hand up stairs, even when the lift is out of order.

## 4 Reason codes
Every missed delivery is logged with result=MISSED and exactly one reason code. The reason code tells the service desk what to fix before rebooking.

- **no-access**: the driver could not get into the building, for example a locked door or a broken door code.
- **bay-closed**: the loading bay was closed or blocked during delivery hours.
- **no-signature**: the driver got in, but nobody was there to sign for the delivery.
- **wrong-address**: the address on the route does not match the client's site.
- **van-breakdown**: the van broke down and the driver could not reach the site.

If more than one reason applies, the driver picks the first one that stopped the delivery. The driver can add a short note, for example "bay closed for building works until Friday".

## 5 Leaving a missed delivery card
The driver leaves a missed delivery card every time a delivery is missed at the site. The card tells the client that we came and how to rebook.

The card shows:

- The date and time of the visit.
- The driver's first name, for example Tom or Aisha.
- The reason in plain words, for example "loading bay closed".
- The service desk details: service@contoso-water.example and 0800 123 4567.

The driver leaves the card at reception, with the building security desk, or in the client's post box. For a van-breakdown, no card is left, because the driver did not reach the site. The service desk emails the client instead.

## 6 Rebooking a missed delivery
The service desk rebooks every missed delivery within 2 working days. The rebooking is done whether or not the client calls.

The service desk:

- Calls or emails the site contact on the same working day the delivery was missed.
- Fixes the cause first, for example gets a new door code or the correct address.
- Updates the access instructions in the client record if anything has changed.
- Books the new delivery on a route within 2 working days and confirms the date by email.

Missed deliveries are not charged. The client pays nothing extra for the missed visit or for the rebooked one.

## 7 The 3-in-a-month rule
When a client has 3 missed deliveries in one calendar month, the account manager calls the client.

All reason codes count towards the 3, including van-breakdown. The service desk tells the account manager as soon as the third missed delivery is logged.

On the call, the account manager:

- Apologises where the cause was on our side.
- Checks the access instructions, opening hours and site contact.
- Agrees a change if needed, for example a new delivery day or a different entrance.

The account manager writes a short summary of the call in the client record. For Fabrikam and Woodgrove, the account manager is Noor Haddad.

## 8 Reporting
Lee Wong reviews missed deliveries every week with the warehouse teams in London, Utrecht and Cologne.

The weekly review looks at the number of missed deliveries, the reason codes, and the routes or clients that repeat. For example, many bay-closed codes on one route may mean the route should start later. Many van-breakdown codes from one office mean the vans need checking.

Missed delivery figures are also part of the monthly complaints review described in SV-CMP-01. Drivers who see the same problem at a site every week should tell their warehouse team lead, so it can be fixed before the next missed delivery.
