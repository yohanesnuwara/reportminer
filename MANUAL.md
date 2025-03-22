MANUAL TO RUN GEODATAX FLASK APP PER 22.03.2025

In Runpod:
1. Deploy pod: GPU RTX A600
2. Click connect, go to SSH, and copy the code

In VS Code: 
3. View > command palette > add new host > paste the SSH code
4. View > command palette > connect to host > pick the IP address
5. New VS code interface will appear

In the SSH-connected VS Code:
6. Open folder > connect repository > input the name of GitHub repository: yohanesnuwara/reportminer
7. Choose the directory: /home
8. Open new terminal 

VS Code terminal:
9. Install all libraries: sh setup.sh
10. Do again: sh setup.sh > click enter to expand the log, answer "yes" when asked, then answer "y"
11. Activate environment: . .venv/bin/activate

Command prompt
12. Open cmd
13. Transfer file from local > runpodctl send "path to folder"
14. Copy the code e.g. runpodctl receive 3222-report-orbit-jackson-4

Back in VS code terminal: 
15. Create new directory inside reportminer: mkdir reports > cd reports
16. Paste the code inside reports folder e.g. runpodctl receive 3222-report-orbit-jackson-4
17. File done transferred
18. Go back to reportminer folder: cd ..
19. Run embedding: python embed.py -i reports # reports is the name of folder 
20. Wait until completed
21. Run app: python app.py
22. Copy the URL e.g. http://127.0.0.1:5000 

Browser: 
23. Paste URL in browser
24. GeoDataX app will appear in browser
