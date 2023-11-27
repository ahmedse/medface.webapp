-- phpMyAdmin SQL Dump
-- version 4.9.5deb2
-- https://www.phpmyadmin.net/
--
-- Host: localhost:3306
-- Generation Time: Nov 13, 2023 at 03:25 PM
-- Server version: 8.0.30-0ubuntu0.20.04.2
-- PHP Version: 7.4.3-4ubuntu2.17

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `medface`
--

-- --------------------------------------------------------

--
-- Table structure for table `medapp_person`
--

CREATE TABLE `medapp_person` (
  `id` int NOT NULL,
  `regno` varchar(15) NOT NULL,
  `fullname` varchar(450) NOT NULL,
  `year` varchar(1) NOT NULL,
  `modifytime` datetime(6) NOT NULL,
  `group` varchar(2) NOT NULL,
  `label` varchar(100) NOT NULL,
  `year_b` varchar(20) NOT NULL,
  `arname` varchar(450) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `medapp_person`
--

INSERT INTO `medapp_person` (`regno`, `fullname`, `year`, `modifytime`, `group`, `label`, `year_b`, `arname`) VALUES
( '20100297', 'Ahmed Yasser', '3', '2023-10-18 10:27:15.114745', '3', '20100297_Ahmed Yasser', '22-26', 'احمد ياسر محمد حسني ابراهيم'),
( '211000007', 'Maryam Tamer', '3', '2023-10-18 10:27:15.203610', '1', '211000007_Maryam Tamer', '22-26', 'مريم تامر نصر محمد'),
( '211000053', 'Rahaf Ibrahim', '3', '2023-10-18 10:27:15.241268', '12', '211000053_Rahaf Ibrahim', '22-26', 'رهف ابراهيم محمد محمود سنجر'),
( '211000072', 'Ahmed Agawany', '3', '2023-10-18 10:27:15.282948', '10', '211000072_Ahmed Agawany', '22-26', 'أحمد محمد أشرف محمود على العجوانى'),
( '211000099', 'Hajer Barakat', '3', '2023-10-18 10:27:15.334247', '10', '211000099_Hajer Barakat', '22-26', 'هاجر حسن بركات حسن'),
( '211000104', 'Menna Elswefy', '3', '2023-10-18 10:27:15.501286', '9', '211000104_Menna Elswefy', '22-26', 'منة الله اشرف علي سويفي بدير'),
( '211000124', 'Retan Hazim', '3', '2023-10-18 10:27:15.551228', '5', '211000124_Retan Hazim', '22-26', 'رتان حازم ابراهيم برانية'),
( '211000137', 'Marwan Elkordy', '3', '2023-10-18 10:27:15.600431', '9', '211000137_Marwan Elkordy', '22-26', 'مروان احمد عبداللطيف الكردي'),
( '211000151', 'Bassant Gamal Eldin', '3', '2023-10-18 10:27:15.634114', '7', '211000151_Bassant Gamal Eldin', '22-26', 'بسنت محمد جمال الدين احمد حسن'),
( '211000157', 'Menna Shamsiya', '3', '2023-10-18 10:27:15.650826', '15', '211000157_Menna Shamsiya', '22-26', 'منة الله ايمن محمد عبده شمسية'),
( '211000266', 'Aya Hisham', '3', '2023-10-18 10:27:15.667534', '15', '211000266_Aya Hisham', '22-26', 'ايه هشام احمد عبدالباقي مصطفى'),
( '211000278', 'Belal Hany', '3', '2023-10-18 10:27:15.701227', '13', '211000278_Belal Hany', '22-26', 'بلال هاني محمد وجدي عبد الفتاح عزالدين'),
( '211000280', 'Rown Adil', '3', '2023-10-18 10:27:15.717043', '14', '211000280_Rown Adil', '22-26', 'روان عادل احمد عادل حفني'),
( '211000370', 'Saif Zaky', '3', '2023-10-18 10:27:15.766875', '2', '211000370_Saif Zaky', '22-26', 'سيف أحمد قاسم زكي'),
( '211000378', 'Reem Hafez', '3', '2023-10-18 10:27:15.817079', '8', '211000378_Reem Hafez', '22-26', 'ريم احمد حافظ عفيفي'),
( '211000406', 'Rana Badr', '3', '2023-10-18 10:27:15.834441', '5', '211000406_Rana Badr', '22-26', 'رنا بدر حسن بدر احمد'),
( '211000408', 'Mohamed Saleh', '3', '2023-10-18 10:27:15.884421', '7', '211000408_Mohamed Saleh', '22-26', 'محمد محمد صالح ثابت'),
( '211000422', 'Hana Elserafy', '3', '2023-10-19 09:36:16.447973', '4', '211000422_Hana Elserafy', '22-26', 'هنا امين جابر محمود الصيرفي'),
( '211000531', 'Haneen Mesbah', '3', '2023-10-18 10:27:15.951311', '13', '211000531_Haneen Mesbah', '22-26', 'حنين محمد السيد مصباح'),
( '211000535', 'Hanaa Islam', '3', '2023-10-18 10:27:15.999846', '13', '211000535_Hanaa Islam', '22-26', 'هنا إسلام حسين علي'),
( '211000539', 'Logyn Attify', '3', '2023-10-18 10:27:16.028929', '11', '211000539_Logyn Attify', '22-26', 'لوجين محمد عطيفي عطيفي مصطفى'),
( '211000547', 'Merna Khalid', '3', '2023-10-18 10:27:16.051009', '1', '211000547_Merna Khalid', '22-26', 'ميرنا خالد عبد الحافظ عبدالله'),
( '211000548', 'Hazim Mostafa', '3', '2023-10-18 10:27:16.270979', '12', '211000548_Hazim Mostafa', '22-26', 'حازم مصطفى سيد احمد عاشور'),
( '211000551', 'Farah Alian', '3', '2023-10-18 10:27:16.333748', '6', '211000551_Farah Alian', '22-26', 'فرح ممدوح سيد احمد عليان'),
( '211000559', 'Alaa Harraz', '3', '2023-10-18 10:27:16.396290', '14', '211000559_Alaa Harraz', '22-26', 'الاء احمد محمد عبد المنعم حراز'),
( '211000593', 'Menna Refaat', '3', '2023-10-18 10:27:16.531674', '11', '211000593_Menna Refaat', '22-26', 'منة الله رفعت محمد محمود'),
( '211000649', 'Nourhan Zanaty', '3', '2023-10-18 10:27:16.615813', '11', '211000649_Nourhan Zanaty', '22-26', 'نورهان محمد محي الدين زناتي احمد'),
( '211000712', 'Mazin Bahnasy', '3', '2023-10-18 10:27:16.757307', '10', '211000712_Mazin Bahnasy', '22-26', 'مازن ياسر السيد بهنسي محمد'),
( '211000718', 'Fredy Emad', '3', '2023-10-18 10:27:16.882829', '9', '211000718_Fredy Emad', '22-26', 'فريدى عماد حنا نسيم'),
( '211000720', 'Ponsieh Samer', '3', '2023-10-18 10:27:16.961325', '15', '211000720_Ponsieh Samer', '22-26', 'بانسيه سامر محمد يسري ابراهيم احمد'),
( '211000767', 'Mahmoud Attian', '3', '2023-10-18 10:27:16.992740', '11', '211000767_Mahmoud Attian', '22-26', 'محمود محمد محمود عطيان'),
( '211000768', 'Renad Ashraf', '3', '2023-10-18 10:27:17.024122', '2', '211000768_Renad Ashraf', '22-26', 'رناد أشرف ابراهيم محمد سعدالدين'),
( '211000778', 'Abdel Rahman Wael', '3', '2023-10-18 10:27:17.040158', '15', '211000778_Abdel Rahman Wael', '22-26', 'عبدالرحمن وائل فاروق عبد الظاهر'),
( '211000784', 'Mohamed Ossama', '3', '2023-10-18 10:27:17.056007', '13', '211000784_Mohamed Ossama', '22-26', 'محمد اسامه محمد صادق شرف'),
( '211000795', 'Omar Amin', '3', '2023-10-18 10:27:17.071420', '2', '211000795_Omar Amin', '22-26', 'عمر امين محمد نافع'),
( '211000798', 'Fliopater Gadallah', '3', '2023-10-18 10:27:17.102670', '14', '211000798_Fliopater Gadallah', '22-26', 'فيلوباتير جادالله مواس جوده جادالله'),
( '211000800', 'Karim Elsamadisy', '3', '2023-10-18 10:27:17.134310', '12', '211000800_Karim Elsamadisy', '22-26', 'كريم احمد موسي السماديسي'),
( '211000809', 'Maryam Elsayed', '3', '2023-10-18 10:27:17.149944', '11', '211000809_Maryam Elsayed', '22-26', 'مريم السيد سعد معتبر احمد'),
( '211000858', 'Nermine Essam', '3', '2023-10-18 10:27:17.181194', '3', '211000858_Nermine Essam', '22-26', 'نرمين عصام الدين على بدوى'),
( '211000864', 'Hazem Elsayed', '3', '2023-10-18 10:27:17.196820', '4', '211000864_Hazem Elsayed', '22-26', 'حازم السيد ابراهيم محمد'),
( '211000869', 'Walid Bahlol', '3', '2023-10-18 10:27:17.244080', '9', '211000869_Walid Bahlol', '22-26', 'وليد أحمد بهلول عبدالعزيز'),
( '211000898', 'Maryam Sabry', '3', '2023-10-18 10:27:17.275337', '1', '211000898_Maryam Sabry', '22-26', 'مريم صبري صلاح الدين عبد الاة'),
( '211000977', 'Mohamed Younis', '3', '2023-10-18 10:27:17.290963', '3', '211000977_Mohamed Younis', '22-26', 'محمد احمد فهمى محمد يونس'),
( '211001167', 'Mostafa Ashraf', '3', '2023-10-18 10:27:17.333255', '8', '211001167_Mostafa Ashraf', '22-26', 'مصطفي اشرف عبدالمعز محمد'),
( '211001178', 'Mohamed Elgazar', '3', '2023-10-18 10:27:17.338262', '7', '211001178_Mohamed Elgazar', '22-26', 'محمد ابراهيم اسماعيل الجزار'),
( '211001436', 'Ahmed Hassan', '3', '2023-10-18 10:27:17.353895', '6', '211001436_Ahmed Hassan', '22-26', 'أحمد حسن محمد خلف'),
( '211001452', 'Heba Ehab', '3', '2023-10-18 10:27:17.400770', '4', '211001452_Heba Ehab', '22-26', 'هبه ايهاب محمد المنشاوي حامد'),
( '211001492', 'Tya Yasser', '3', '2023-10-18 10:27:17.433525', '4', '211001492_Tya Yasser', '22-26', 'تيا ياسر على مصلح صلاح'),
( '211001669', 'Mina Louise', '3', '2023-10-18 10:27:17.448031', '14', '211001669_Mina Louise', '22-26', 'مينا لويز فتحي نظير بطرس'),
( '211001676', 'Kerolous Youssef', '3', '2023-10-18 10:27:17.479289', '15', '211001676_Kerolous Youssef', '22-26', 'كيرلس يوسف حنا ايوب'),
( '211001707', 'Karim Hashim', '3', '2023-10-18 10:27:17.510571', '10', '211001707_Karim Hashim', '22-26', 'كريم محمد حمدان هاشم'),
( '211001708', 'Hana Hesham', '3', '2023-10-18 10:27:17.554765', '1', '211001708_Hana Hesham', '22-26', 'هنا هشام رشاد محمد عاشور'),
( '211001914', 'Arwah Ahmed', '3', '2023-10-18 10:27:17.571260', '4', '211001914_Arwah Ahmed', '22-26', 'أروى احمد محمد حلمي عبدالعزيز'),
( '211001921', 'Maya Abdo', '3', '2023-10-18 10:27:17.586935', '13', '211001921_Maya Abdo', '22-26', 'مايا احمد محمد عبده'),
( '211001927', 'Hanin Zahran', '3', '2023-10-18 10:27:17.633696', '16', '211001927_Hanin Zahran', '22-26', 'حنين زهران احمد زهران دسوقي'),
( '211002119', 'Nada Ashoush', '3', '2023-10-18 10:27:17.649376', '8', '211002119_Nada Ashoush', '22-26', 'ندى مصطفى عبدالمنعم محمد محمد عشوش'),
( '211002262', 'Yousef Imam', '3', '2023-10-18 10:27:17.680624', '11', '211002262_Yousef Imam', '22-26', 'يوسف على احمد على الامام'),
( '211002299', 'Zayneb Soliman', '3', '2023-10-18 10:27:17.711875', '3', '211002299_Zayneb Soliman', '22-26', 'زينب سليمان عبد العظيم أحمد'),
( '211002333', 'Jasmin Hassan', '3', '2023-10-18 10:27:17.774876', '7', '211002333_Jasmin Hassan', '22-26', 'جاسمين حسن صابر رمضان محمد'),
( '211002432', 'Youssef Sherif', '3', '2023-10-18 10:27:17.821925', '16', '211002432_Youssef Sherif', '22-26', 'يوسف شريف محمد عبدالعزيز حسن'),
( '211002523', 'Moaz Abdel Raouf', '3', '2023-10-18 10:27:17.854030', '4', '211002523_Moaz Abdel Raouf', '22-26', 'معاذ علاء عبد الروف مبروك'),
( '211002588', 'Ahmed Abdel Moez', '3', '2023-10-18 10:27:17.869625', '12', '211002588_Ahmed Abdel Moez', '22-26', 'احمد محمد عبد المعز محمود على'),
( '211002591', 'Aliaa Hablas', '3', '2023-10-18 10:27:17.900694', '1', '211002591_Aliaa Hablas', '22-26', 'علياء احمد محمود حبلص'),
( '211002656', 'Nabil Oraby', '3', '2023-10-18 10:27:17.933953', '2', '211002656_Nabil Oraby', '22-26', 'نبيل يوسف محمد يوسف'),
( '211002669', 'Ranah Elsayed', '3', '2023-10-18 10:27:17.947964', '1', '211002669_Ranah Elsayed', '22-26', 'رنا السيد علي السيد'),
( '211002693', 'Alaa Hassan', '3', '2023-10-18 10:27:17.979492', '6', '211002693_Alaa Hassan', '22-26', 'علاء حسن عبدالجواد علي عثمان'),
( '211002699', 'Karim Shaalan', '3', '2023-10-18 10:27:18.010488', '8', '211002699_Karim Shaalan', '22-26', 'كريم محمد السيد عبد الغفار شعلان'),
( '211002717', 'Heba Hammam', '3', '2023-10-18 10:27:18.089347', '9', '211002717_Heba Hammam', '22-26', 'هبه نبيل همام السيد عمر'),
( '211002721', 'Bassmala El Hashimy', '3', '2023-10-18 10:27:18.199385', '3', '211002721_Bassmala El Hashimy', '22-26', 'بسملة محمد الهاشمي ابراهيم'),
( '211002785', 'Amal Arafa', '3', '2023-10-18 10:27:18.215009', '7', '211002785_Amal Arafa', '22-26', 'أمل محمد محمد عرفة'),
( '211002789', 'Lama Galal', '3', '2023-10-18 10:27:18.262435', '12', '211002789_Lama Galal', '22-26', 'لمي جلال محمد الطاهر'),
( '211002967', 'Solan Amgad', '3', '2023-10-18 10:27:18.324941', '5', '211002967_Solan Amgad', '22-26', 'سولان امجد ابراهيم خليل'),
( '211002984', 'Abdallah Emad', '3', '2023-10-18 10:27:18.341230', '13', '211002984_Abdallah Emad', '22-26', 'عبدالله عماد على محمد'),
( '211002997', 'Ahmed Attiah', '3', '2023-10-18 10:27:18.403967', '16', '211002997_Ahmed Attiah', '22-26', 'أحمد عطيه عبدالحليم عبد اللطيف نبيوه'),
( '211003011', 'Menna Sabry', '3', '2023-10-18 10:27:18.419560', '6', '211003011_Menna Sabry', '22-26', 'منة الله صبري محمد عبدالكريم حسين'),
( '211003023', 'Jannah Walid', '3', '2023-10-18 10:27:18.435733', '13', '211003023_Jannah Walid', '22-26', 'جنة وليد السيد سالم بدوى'),
( '211003042', 'Mahmoud Ossama', '3', '2023-10-18 10:27:18.451372', '16', '211003042_Mahmoud Ossama', '22-26', 'محمود اسامه محمود محمد يوسف'),
( '211003094', 'Asser Mahmoud', '3', '2023-10-18 10:27:18.467263', '11', '211003094_Asser Mahmoud', '22-26', 'أسر محمود عبدالعزيز الابياري'),
( '211003183', 'Donia Mohamed', '3', '2023-10-18 10:27:18.482888', '3', '211003183_Donia Mohamed', '22-26', 'دنيا محمد سعد عبدالمقصود'),
( '211003211', 'Youssef Tantawy', '3', '2023-10-18 10:27:18.498438', '4', '211003211_Youssef Tantawy', '22-26', 'يوسف حسين عزب طنطاوي'),
( '211003252', 'Sherif Ashraf', '3', '2023-10-18 10:27:18.514139', '8', '211003252_Sherif Ashraf', '22-26', 'شريف اشرف اسماعيل محمد هنداوى'),
( '211003335', 'Abdallah Saied', '3', '2023-10-18 10:27:18.534565', '6', '211003335_Abdallah Saied', '22-26', 'عبدالله سعيد عبدالفضيل عبدالعزيز'),
( '211003336', 'Sherihan Hassan', '3', '2023-10-18 10:27:18.577265', '5', '211003336_Sherihan Hassan', '22-26', 'شريهان حسن مسعد ابو المجد البسيوني'),
( '211003418', 'Ehab Saber', '3', '2023-10-18 10:27:18.608514', '8', '211003418_Ehab Saber', '22-26', 'ايهاب صابر محمد الشركسي'),
( '211003470', 'Safa Alaa', '3', '2023-10-18 10:27:18.624141', '16', '211003470_Safa Alaa', '22-26', 'صفا علاء منصور احمد'),
( '211003497', 'Abdel Rahman Hegazy', '3', '2023-10-18 10:27:18.640430', '12', '211003497_Abdel Rahman Hegazy', '22-26', 'عبد الرحمن حجازي محمد حسن صالح'),
( '211003523', 'Ahmed Khamis', '3', '2023-10-18 10:27:18.656333', '11', '211003523_Ahmed Khamis', '22-26', 'احمد خميس سعد ابوحسن'),
( '211003548', 'Janna Younis', '3', '2023-10-18 10:27:18.671960', '6', '211003548_Janna Younis', '22-26', 'جنه محمد احمد يونس'),
( '211003626', 'Abdallah Salman', '3', '2023-10-18 10:27:18.687542', '3', '211003626_Abdallah Salman', '22-26', 'عبدالله سعيد عبدالفتاح سلمان'),
( '211003629', 'Sama Yasser', '3', '2023-10-18 10:27:18.703129', '1', '211003629_Sama Yasser', '22-26', 'سما ياسر احمد زايد'),
( '211003630', 'Yassmin Kandil', '3', '2023-10-18 10:27:18.718606', '8', '211003630_Yassmin Kandil', '22-26', 'ياسمين صلاح محمد قنديل'),
( '211003660', 'Habiba Mohamed', '3', '2023-10-18 10:27:18.782268', '16', '211003660_Habiba Mohamed', '22-26', 'حبيبه محمد حماده شديد احمد نور الدين'),
( '211003661', 'Jessica Mashally', '3', '2023-10-18 10:27:18.813517', '7', '211003661_Jessica Mashally', '22-26', 'جاسيكا ميشيل جوزيف جبرة'),
( '211003707', 'Abdellatif Ahmed', '3', '2023-10-18 10:27:18.835167', '4', '211003707_Abdellatif Ahmed', '22-26', 'عبد اللطيف احمد عبد اللطيف'),
( '211003737', 'Manar Shahin', '3', '2023-10-18 10:27:18.845181', '12', '211003737_Manar Shahin', '22-26', 'منار عماد فوزي شاهين'),
( '211003745', 'Ahmed Medhat', '3', '2023-10-18 10:27:18.861079', '10', '211003745_Ahmed Medhat', '22-26', 'أحمد مدحت محمد محمد أبو الفضل'),
( '211003760', 'Sarah El Kerdawi', '3', '2023-10-18 10:27:18.876706', '2', '211003760_Sarah El Kerdawi', '22-26', 'سارة محمد محمود الكرداوي'),
( '211003768', 'Wesal Diyaa', '3', '2023-10-18 10:27:18.892112', '13', '211003768_Wesal Diyaa', '22-26', 'وصال ضياء سليمان محمد'),
( '211003775', 'Ammar Yasser', '3', '2023-10-18 10:27:18.907958', '7', '211003775_Ammar Yasser', '22-26', 'عمار ياسر عبد المجيد مصطفى العشري'),
( '211003889', 'Mahmoud Yehia', '3', '2023-10-18 10:27:18.923535', '1', '211003889_Mahmoud Yehia', '22-26', 'محمود يحيى محمود باشا'),
( '211007008', 'Youssef Ettman', '3', '2023-10-18 10:27:18.939784', '2', '211007008_Youssef Ettman', '22-26', 'يوسف عبد الله ابراهيم عتمان'),
( '211008086', 'Nour Mohsen', '3', '2023-10-18 10:27:18.955422', '16', '211008086_Nour Mohsen', '22-26', 'نور محسن احمد علي أبوشوشة'),
( '211008088', 'Rana Yasser', '3', '2023-10-18 10:27:18.986712', '2', '211008088_Rana Yasser', '22-26', 'رنا ياسر ابراهيم ابراهيم قلموش'),
( '211008096', 'Omar Ossama', '3', '2023-10-18 10:27:18.986712', '15', '211008096_Omar Ossama', '22-26', 'عمر اسامه حسين عبدالحميد'),
( '211008694', 'Sarah Harmoush', '3', '2023-10-18 10:27:19.018197', '5', '211008694_Sarah Harmoush', '22-26', 'ساره مصطفى محمد حرموش'),
( '211008759', 'Sief Atta', '3', '2023-10-18 10:27:19.033814', '3', '211008759_Sief Atta', '22-26', 'سيف الدين عطا محي الدين عطا السيد'),
( '211008767', 'Merna Nasser', '3', '2023-10-18 10:27:19.066043', '11', '211008767_Merna Nasser', '22-26', 'ميرنا ناصر أحمد عبد الله البهنساوي'),
( '211008839', 'Mostafa Serah', '3', '2023-10-18 10:27:19.097294', '5', '211008839_Mostafa Serah', '22-26', 'مصطفى أحمد محمد صيره'),
( '211008843', 'John Nabil', '3', '2023-10-18 10:27:19.112921', '5', '211008843_John Nabil', '22-26', 'جون نبيل ميخائيل وديع'),
( '211008964', 'Ahmed Elfeel', '3', '2023-10-18 10:27:19.128545', '12', '211008964_Ahmed Elfeel', '22-26', 'احمد علاء عبد العظيم محمد الفيل'),
( '211009356', 'Mostafa Abdel Halim', '3', '2023-10-18 10:27:19.191982', '6', '211009356_Mostafa Abdel Halim', '22-26', 'مصطفى محمد عبد الحليم ابراهيم يوسف'),
( '211010040', 'Zainab Ibrahim', '3', '2023-10-18 10:27:19.239522', '3', '211010040_Zainab Ibrahim', '22-26', 'زينب ابراهيم ابراهيم سيد أحمد سويدان'),
( '211010102', 'Fatma Elzahraa Emad', '3', '2023-10-18 10:27:19.255320', '10', '211010102_Fatma Elzahraa Emad', '22-26', 'فاطمة الزهراء عماد حمدى منسى'),
( '211010129', 'Mennah Mostafa', '3', '2023-10-18 10:27:19.286445', '4', '211010129_Mennah Mostafa', '22-26', 'منة الله مصطفى عبدرالحمن حسن'),
( '211010161', 'Salah Koriem', '3', '2023-10-18 10:27:19.333560', '9', '211010161_Salah Koriem', '22-26', 'صلاح اسامه عبدالله كريم'),
( '211010329', 'Antony Mamdouh', '3', '2023-10-18 10:27:19.349854', '9', '211010329_Antony Mamdouh', '22-26', 'انتوني ممدوح اديب غطاس ميخائيل'),
( '211010358', 'Arwah Saleh', '3', '2023-10-18 10:27:19.365758', '1', '211010358_Arwah Saleh', '22-26', 'أروى محمد صالح ثابت عبد الله'),
( '211010553', 'karim Mohsen', '3', '2023-10-18 10:27:19.381391', '7', '211010553_karim Mohsen', '22-26', 'كريم محسن محمد محمد الخياط'),
( '211010638', 'Mohamed Nazieh', '3', '2023-10-18 10:27:19.412639', '7', '211010638_Mohamed Nazieh', '22-26', 'محمد نزيه معوض على ابراهيم'),
( '211010703', 'Omnia Shaaban', '3', '2023-10-18 10:27:19.444233', '10', '211010703_Omnia Shaaban', '22-26', 'امنيه شعبان محمد الفقي'),
( '211010731', 'Nouran Khaled', '3', '2023-10-18 10:27:19.459906', '8', '211010731_Nouran Khaled', '22-26', 'نوران خالد محمد الحصري'),
( '211010750', 'Faris Rabie', '3', '2023-10-18 10:27:19.475768', '12', '211010750_Faris Rabie', '22-26', 'فارس محمد ربيع'),
( '211010753', 'Yassin Kafrawy', '3', '2023-10-18 10:27:19.507020', '14', '211010753_Yassin Kafrawy', '22-26', 'ياسين نبيل عبد الفتاح الكفراوى'),
( '211010755', 'Arsany Fawzy', '3', '2023-10-18 10:27:19.554228', '6', '211010755_Arsany Fawzy', '22-26', 'ارساني فوزي عبدة شاكر'),
( '211010773', 'Ashrakat Hussien', '3', '2023-10-18 10:27:19.585442', '9', '211010773_Ashrakat Hussien', '22-26', 'اشرقت على السيد حسين'),
( '211010811', 'Heba Elsayem', '3', '2023-10-18 10:27:19.616692', '13', '211010811_Heba Elsayem', '22-26', 'هبه الصايم شعبان محمد'),
( '211010842', 'Eyad Gamal', '3', '2023-10-18 10:27:19.632318', '10', '211010842_Eyad Gamal', '22-26', 'اياد جمال عبد الله احمد المغربي'),
( '211010869', 'Abd ElMasih Sabry', '3', '2023-10-18 10:27:19.652908', '16', '211010869_Abd ElMasih Sabry', '22-26', 'عبدالمسيح صبري سعد جندي'),
( '211010944', 'Kirollos William', '3', '2023-10-18 10:27:19.657040', '10', '211010944_Kirollos William', '22-26', 'كيرلس وليم خلف وهيب'),
( '211010960', 'Saleh Mohamed', '3', '2023-10-18 10:27:19.672669', '4', '211010960_Saleh Mohamed', '22-26', 'صالح محمد حسن علي'),
( '211010967', 'Abdel Rahman El Hasafy', '3', '2023-10-18 10:27:19.688295', '12', '211010967_Abdel Rahman El Hasafy', '22-26', 'عبدالرحمن محمد الحصافي ابو عياشة'),
( '211010968', 'Youssef Ali', '3', '2023-10-18 10:27:19.719550', '11', '211010968_Youssef Ali', '22-26', 'يوسف على محمد على أحمد'),
( '211010973', 'Zina Fayez', '3', '2023-10-18 10:27:19.737214', '14', '211010973_Zina Fayez', '22-26', 'زينه محمد فايز المصري'),
( '211011032', 'Abdel Rahman Farid', '3', '2023-10-18 10:27:19.760555', '8', '211011032_Abdel Rahman Farid', '22-26', 'عبدالرحمن فريد محمود المدني'),
( '211011035', 'Abdelrahman Seddik', '3', '2023-10-18 10:27:19.802618', '5', '211011035_Abdelrahman Seddik', '22-26', 'عبدالرحمن صديق عطية الجمل'),
( '211011044', 'Bassem Nader', '3', '2023-10-18 10:27:19.808399', '6', '211011044_Bassem Nader', '22-26', 'باسم نادر حسين فؤاد'),
( '211011070', 'Jana Ossama', '3', '2023-10-18 10:27:19.835916', '15', '211011070_Jana Ossama', '22-26', 'جني اسامه على عبد اللطيف محمد'),
( '211011097', 'Maryam Shaltout', '3', '2023-10-18 10:27:19.838996', '5', '211011097_Maryam Shaltout', '22-26', 'مريم محمد سالم محمد شلتوت'),
( '211011119', 'Khalid Essam', '3', '2023-10-18 10:27:19.870327', '5', '211011119_Khalid Essam', '22-26', 'خالد عصام احمد انور محمد سليم'),
( '211011145', 'Abdel Rahman Saniour', '3', '2023-10-18 10:27:19.873423', '6', '211011145_Abdel Rahman Saniour', '22-26', 'عبدالرحمن علي حسن محمد سنيور'),
( '211011152', 'Abdelrahman Ashour', '3', '2023-10-18 10:27:19.902625', '14', '211011152_Abdelrahman Ashour', '22-26', 'عبدالرحمن عاشور عبدالرحمن علي'),
( '211011153', 'Ahmed Gabr', '3', '2023-10-18 10:27:19.917632', '15', '211011153_Ahmed Gabr', '22-26', 'احمد فتحي جبر علي مصطفى'),
( '211011174', 'Baraa Hamdy', '3', '2023-10-18 10:27:19.935636', '9', '211011174_Baraa Hamdy', '22-26', 'براء حمدى السيد عسقلانى'),
( '211011177', 'Bassmalah Farag', '3', '2023-10-18 10:27:19.951149', '8', '211011177_Bassmalah Farag', '22-26', 'بسمله فرج السيد محمود غنيم'),
( '211011222', 'Omar El Dakkak', '3', '2023-10-18 10:27:19.955149', '1', '211011222_Omar El Dakkak', '22-26', 'عمر احمد حسن'),
( '211011225', 'Ahmed Sherief', '3', '2023-10-18 10:27:20.020667', '14', '211011225_Ahmed Sherief', '22-26', 'احمد شريف أحمد خميس'),
( '211011237', 'Reeman Elsayed', '3', '2023-10-18 10:27:20.035699', '9', '211011237_Reeman Elsayed', '22-26', 'ريمان محمد على السيد'),
( '211011260', 'Janah Shady', '3', '2023-10-18 10:27:20.038234', '2', '211011260_Janah Shady', '22-26', 'جنى شادى يوسف ناصف'),
( '211011261', 'Lougin Shady', '3', '2023-10-18 10:27:20.068920', '2', '211011261_Lougin Shady', '22-26', 'لجين شادى يوسف ناصف'),
( '211011273', 'Shahd Roushdy', '3', '2023-10-18 10:27:20.083950', '14', '211011273_Shahd Roushdy', '22-26', 'شهد محمد رشدي'),
( '211011276', 'Abdallah Hisham', '3', '2023-10-18 10:27:20.102714', '10', '211011276_Abdallah Hisham', '22-26', 'عبد الله هشام سعد السيد'),
( '211012090', 'Magda Mohamed', '3', '2023-10-18 10:27:20.117736', '4', '211012090_Magda Mohamed', '22-26', 'ماجدة محمد السيد'),
( '212000100', 'Adham Ahmed', '3', '2023-10-18 10:27:20.153532', '13', '212000100_Adham Ahmed', '22-26', 'ادهم احمد ابراهيم سعيد'),
( '212001003', 'Abdelrahamn Barbary', '3', '2023-10-18 10:27:20.169159', '11', '212001003_Abdelrahamn Barbary', '22-26', 'عبدالرحمن سامي البربري'),
( '212001020', 'Hamza Badawy', '3', '2023-10-18 10:27:20.184784', '14', '212001020_Hamza Badawy', '22-26', 'حمزة احمد بدوي عبدالفتاح'),
( '212001034', 'Mohamed Sabry', '3', '2023-10-18 10:27:20.200409', '15', '212001034_Mohamed Sabry', '22-26', 'محمد صبري يس ابراهيم'),
( '212001074', 'Ahmed Ibrahim', '3', '2023-10-18 10:27:20.216035', '1', '212001074_Ahmed Ibrahim', '22-26', 'أحمد محمد ابراهيم السيد عبد الحي'),
( '212001084', 'Assem Mohamed', '3', '2023-10-18 10:27:20.237695', '3', '212001084_Assem Mohamed', '22-26', 'عاصم محمد محمود عبدالعليم'),
( '212001107', 'Islam Alaa Eldin', '3', '2023-10-18 10:27:20.279021', '11', '212001107_Islam Alaa Eldin', '22-26', 'إسلام علاءالدين محمود احمد'),
( '212001112', 'Maryam Hammam', '3', '2023-10-18 10:27:20.311108', '6', '212001112_Maryam Hammam', '22-26', 'مريم علاء همام على الدين'),
( '212001131', 'Ahmed El ashry', '3', '2023-10-18 10:27:20.321245', '8', '212001131_Ahmed El ashry', '22-26', 'احمد اسامة عبدالغني العشري'),
( '212001215', 'Sara Alaa', '3', '2023-10-18 10:27:20.337929', '9', '212001215_Sara Alaa', '22-26', 'ساره علاء ابو النصر شهاب'),
( '212001217', 'Ibrahim Elmahdy', '3', '2023-10-18 10:27:20.352968', '10', '212001217_Ibrahim Elmahdy', '22-26', 'ابراهيم أحمد حازم المهدي'),
( '212001270', 'Lina Sameh', '3', '2023-10-18 10:27:20.368631', '13', '212001270_Lina Sameh', '22-26', 'لينة سامح إبراهيم الشناوي'),
( '212001278', 'Hamza Tawfik', '3', '2023-10-18 10:27:20.384257', '12', '212001278_Hamza Tawfik', '22-26', 'حمزه خالد محمد توفيق'),
( '212001377', 'Mohamed Hatem', '3', '2023-10-18 10:27:20.447206', '15', '212001377_Mohamed Hatem', '22-26', 'محمد حاتم حسن'),
( '212001409', 'Enjy Emad', '3', '2023-10-18 10:27:20.462868', '2', '212001409_Enjy Emad', '22-26', 'انجي عماد عبدالحميد المغربي'),
( '221003615', 'Abdallah ElShalkany', '3', '2023-10-18 10:27:20.509744', '16', '221003615_Abdallah ElShalkany', '22-26', 'عبد الله عمرو محمد الشلقاني'),
( '231001961', 'Amr Hablas', '3', '2023-10-18 10:27:20.525371', '16', '231001961_Amr Hablas', '22-26', 'عمرو أحمد محمود عبدالله حبلص');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `medapp_person`
--
ALTER TABLE `medapp_person`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `regno` (`regno`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `medapp_person`
--
ALTER TABLE `medapp_person`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=170;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
