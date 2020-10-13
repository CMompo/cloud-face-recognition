FROM mariadb:latest

ENV MYSQL_ROOT_PASSWORD=4yXNPkrc \
	MYSQL_USER=kub \
	MYSQL_PASSWORD=4yXNPkrc \
	MYSQL_DATABASE=face-recognition-db

COPY db-gen/face_recognition_db.sql /docker-entrypoint-initdb.d/face_recognition_db.sql