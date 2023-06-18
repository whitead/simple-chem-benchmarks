from csv import excel_tab
import datasets
from ord_schema.proto import dataset_pb2, reaction_pb2
import urllib.request
import gzip

_URL = "https://github.com/open-reaction-database/ord-data/raw/main/data/"
_FILES = [
    "00/ord_dataset-00005539a1e04c809a9a78647bea649c.pb.gz",
    "01/ord_dataset-018fd0e1351f4fd09b20fcddd97b4c7a.pb.gz",
    "01/ord_dataset-01dbb772c5e249108f0b191ed17a2c0c.pb.gz",
    "02/ord_dataset-02ee2261663048188cf6d85d2cc96e3f.pb.gz",
    "03/ord_dataset-0387783899c642a8b7eb4ba379bcdf5d.pb.gz",
    "03/ord_dataset-03ba810b7f464a06b5d8787af2e8b64e.pb.gz",
    "04/ord_dataset-04982f13ed08448d93df6794846500f3.pb.gz",
    "04/ord_dataset-04d607efe1d9485eb99fafa06880f62e.pb.gz",
    "05/ord_dataset-053897d9b1744303b2fdfe3e796ca27b.pb.gz",
    "06/ord_dataset-06d4002fc4d34860a0688cba690e12dc.pb.gz",
    "07/ord_dataset-074f86301ec5441ab3b52d902ac06949.pb.gz",
    "07/ord_dataset-07db50a3ce6941919df30a9e2898988f.pb.gz",
    "08/ord_dataset-081613ef79bd4110aacc146b4465f086.pb.gz",
    "08/ord_dataset-08852243bba44cb28769a5833f1515fe.pb.gz",
    "09/ord_dataset-09e9a37ee5794dc28c0ad7bf7a442c18.pb.gz",
    "0a/ord_dataset-0a66204fc43e49c2922e6f9107e6b62f.pb.gz",
    "0b/ord_dataset-0b24d1f58a024ed7bc2bda95b2430c72.pb.gz",
    "0b/ord_dataset-0b32b90cc77b4a3db47ad263e0bbc1a8.pb.gz",
    "0b/ord_dataset-0b70410902ae4139bd5d334881938f69.pb.gz",
    "0b/ord_dataset-0bb2e99daa66408fb8dbd6a0781d241c.pb.gz",
    "0b/ord_dataset-0bf72e95d80743729fdbb8b57a4bc0c6.pb.gz",
    "0c/ord_dataset-0c37e633e9814a6e886187796cacf216.pb.gz",
    "0c/ord_dataset-0c61835e3a0b4986aabf2b61b708e322.pb.gz",
    "0c/ord_dataset-0c75d67751634f0594b24b9f498b77c2.pb.gz",
    "0c/ord_dataset-0ca5627a13c049a99463095023b09fe5.pb.gz",
    "0f/ord_dataset-0f9d2dbe929a45c3892ae75e81e99443.pb.gz",
    "10/ord_dataset-10360c60962940178e4bf05e54b1655c.pb.gz",
    "10/ord_dataset-10b940e7982c4622b1e1ac879394aba6.pb.gz",
    "10/ord_dataset-10f2568b472a4cabb35c3f313a159005.pb.gz",
    "11/ord_dataset-11068b1e475b4c5b82e56726ab0dddb7.pb.gz",
    "11/ord_dataset-11b3c3b41eda49e196ec983a65d3b2c0.pb.gz",
    "12/ord_dataset-127eb5fdd5b4402e92b51d0b55a5dcb5.pb.gz",
    "12/ord_dataset-128a4f3f44e844218b1c319ab4de3893.pb.gz",
    "12/ord_dataset-12a0ec4f3cda46e2badc694da2072c39.pb.gz",
    "12/ord_dataset-12dc3bd21bcf44d09e5b4249afe15161.pb.gz",
    "12/ord_dataset-12ef86aced0149e0917be82ce22190e2.pb.gz",
    "13/ord_dataset-134cf2fa32ab464880d75db06c38f35a.pb.gz",
    "13/ord_dataset-136cfada6ce247b4919085a57363459e.pb.gz",
    "13/ord_dataset-1385ebf1988241e49636101695ad79e4.pb.gz",
    "14/ord_dataset-140fb5527ff24f97bcf0094c5d100120.pb.gz",
    "14/ord_dataset-14866e68bb5b47f5929457eecb4eb3a5.pb.gz",
    "15/ord_dataset-1558660634294cc8ad7e01746e9083fd.pb.gz",
    "15/ord_dataset-1590b34936164499aa4ed4fe0ef21255.pb.gz",
    "15/ord_dataset-159c363342e44b539e7a5975c873e30d.pb.gz",
    "15/ord_dataset-15cdba4c7f064b3f9cd7343cb3187881.pb.gz",
    "15/ord_dataset-15ce1bcfb62046d9bec87d32620888d5.pb.gz",
    "17/ord_dataset-171b840ae6e84e45bab43b987d09f5c7.pb.gz",
    "18/ord_dataset-180e296d6d6245638d4d22a59120ea01.pb.gz",
    "18/ord_dataset-1884c7bf3d544afdb8d17b5d41b90a27.pb.gz",
    "18/ord_dataset-1895fe091c3f47afa1ee96a41a250de4.pb.gz",
    "18/ord_dataset-18e9ed24dbd44e98b33bdc22aa7580a8.pb.gz",
    "19/ord_dataset-19041548671f410397ce7f00536a9be3.pb.gz",
    "19/ord_dataset-1953a9402dda47b6bcfe3e3cb9ab8a07.pb.gz",
    "19/ord_dataset-19b9e29d593f4660ab77c07b459da9bb.pb.gz",
    "19/ord_dataset-19e5fc80c1554f4f8641c835e055f02b.pb.gz",
    "1a/ord_dataset-1a1aa5d1c3224edca0aec6e3398da985.pb.gz",
    "1a/ord_dataset-1a231de00bfe4443b547e1f03885ed41.pb.gz",
    "1a/ord_dataset-1acb071a357f438ea5993287375971cf.pb.gz",
    "1b/ord_dataset-1b0cd79134f0450eaac8396a4f956c30.pb.gz",
    "1c/ord_dataset-1c0bae7388cf460091d56129e95b3145.pb.gz",
    "1c/ord_dataset-1c61a371eaf841328cc190a0356d6b3b.pb.gz",
    "1c/ord_dataset-1cb9d78632144c5f8acfc3e9ff388678.pb.gz",
    "1d/ord_dataset-1d493fcbd5494078b46e816998dcb991.pb.gz",
    "1d/ord_dataset-1d5bba1436bd42b7a27c43833c25d871.pb.gz",
    "1e/ord_dataset-1ecf96d88f254270bff816ee7eeffef6.pb.gz",
    "20/ord_dataset-2069a3db775a4d9f9310aca092ecbad8.pb.gz",
    "21/ord_dataset-21c1b1c06c7e4e09a38b5b1c71a32e52.pb.gz",
    "23/ord_dataset-232d09663ed349c2a1fb1f05d411eae0.pb.gz",
    "23/ord_dataset-2345345e84034af393f969fc5e7741b6.pb.gz",
    "23/ord_dataset-2369b9b9f44641b1930518c59ae89a95.pb.gz",
    "24/ord_dataset-24ad29d930104afea313b6c3bd11099e.pb.gz",
    "25/ord_dataset-2501595af7f242268dbaab34c9e7ae56.pb.gz",
    "26/ord_dataset-262b40ea420c471da9b9244fe9b8f645.pb.gz",
    "26/ord_dataset-266f60b4555945d6afb2d2bdf5fa04e0.pb.gz",
    "27/ord_dataset-271c0b74f4794a06992957029b3151ba.pb.gz",
    "27/ord_dataset-273fda773e864aaf9b71a30a2d9f2162.pb.gz",
    "27/ord_dataset-275344fd078b4340b89ca0b6e92beb95.pb.gz",
    "27/ord_dataset-275a3da8f45f4536ad29727f0ef9ba66.pb.gz",
    "27/ord_dataset-278fb6af8a994ac69e4d95e6e6eff756.pb.gz",
    "28/ord_dataset-2841bde0239a4964a69f490c014a6e43.pb.gz",
    "28/ord_dataset-285df12e34cd46e993e3c8ebc3a8962a.pb.gz",
    "28/ord_dataset-28b19b59e93c47698a5a2b21048ec201.pb.gz",
    "29/ord_dataset-293186f5c9b441cab57f03cd3a18ac26.pb.gz",
    "29/ord_dataset-2952e63264f5422a84e12cca1e0541ee.pb.gz",
    "29/ord_dataset-29d79fca4cec4a43b773d0ba25b27651.pb.gz",
    "2a/ord_dataset-2a1960001e7d4e89987253631df1362a.pb.gz",
    "2a/ord_dataset-2ada3fda46fc44719ba0c8001f53c1b3.pb.gz",
    "2c/ord_dataset-2c460e2ef9934444aaf26fec1f75741f.pb.gz",
    "2c/ord_dataset-2cb52357eb5f43c2be56ad5c0662f18f.pb.gz",
    "2d/ord_dataset-2d21ceeb77844edc9eaedb2fabca2ff1.pb.gz",
    "2d/ord_dataset-2d589ad46f82417ab9ddc07f7655411c.pb.gz",
    "2d/ord_dataset-2d6edb8ffd434003bb508360153bd9bb.pb.gz",
    "2e/ord_dataset-2e58cb8db2bf482bbea23283b7e04488.pb.gz",
    "2e/ord_dataset-2e96d163c3f64eb5b470e1ea1a41922a.pb.gz",
    "2f/ord_dataset-2f37329a4b254471a74f2eb0981f11ec.pb.gz",
    "30/ord_dataset-3007013966624a1096d71142f31cb5a3.pb.gz",
    "30/ord_dataset-30ad5cf6083a45a387b45bebad1a4d65.pb.gz",
    "31/ord_dataset-31641fb65b34430fa7435229b949b604.pb.gz",
    "31/ord_dataset-31f00a039ca0424788e5e1970d25a8fd.pb.gz",
    "31/ord_dataset-31fc6d0085ca4d8dbbcd3a5fa9dcedfb.pb.gz",
    "32/ord_dataset-324fb6fdc2414cb79e436bf5d04d4bd2.pb.gz",
    "34/ord_dataset-343f84c5bd164b25aeae96e1d828e768.pb.gz",
    "34/ord_dataset-347c0709d28a44dea43ca42052be4db3.pb.gz",
    "34/ord_dataset-34cf5f1378de4e34844aea93a2f9a9d3.pb.gz",
    "35/ord_dataset-3577d334f6eb4dc4bd73564fee3f0dfc.pb.gz",
    "35/ord_dataset-35824232b132464aa99e71aba765981d.pb.gz",
    "35/ord_dataset-3592bd645cd143ee8274cd0d834ae581.pb.gz",
    "35/ord_dataset-359b8fc87f4244be89d6f02bc5036eac.pb.gz",
    "35/ord_dataset-35a5a513f1dd44a3a97c88da99f81a00.pb.gz",
    "35/ord_dataset-35b56288528641309a040cc2b6710b61.pb.gz",
    "35/ord_dataset-35c51552812941cda45194a013d34bb9.pb.gz",
    "36/ord_dataset-36057d699ac5449e9c37eb99abf78b03.pb.gz",
    "36/ord_dataset-366bdd9ee72d474cbe6f3f9e54dd96d2.pb.gz",
    "36/ord_dataset-36f7bef430f14c2e8db5e3f7c3640d9b.pb.gz",
    "37/ord_dataset-3708161f4ba04e959b9a7a8d59fd86e1.pb.gz",
    "37/ord_dataset-373415d3e0e54004837cf4831e67666f.pb.gz",
    "37/ord_dataset-375a420ee9b042918ddca20f02df37d3.pb.gz",
    "37/ord_dataset-37b0416f244344a08cf357e851eedf2a.pb.gz",
    "37/ord_dataset-37d3220f708c49ad839bab296b722248.pb.gz",
    "38/ord_dataset-380e279f82154dba9e08ab51b3bdd08a.pb.gz",
    "38/ord_dataset-3844acbccc714c04ab757ec4fca10bd0.pb.gz",
    "38/ord_dataset-386da077ab2340638cada986e2ef0770.pb.gz",
    "39/ord_dataset-3947f3e1be17462c8f7c7e6ea6e57d0a.pb.gz",
    "39/ord_dataset-3953983e052a4076aa7cc0880b79cb8b.pb.gz",
    "39/ord_dataset-39f318aa6ec5450182abaf3662294306.pb.gz",
    "3a/ord_dataset-3af92aec23dc4810b92eb0d8c60023ee.pb.gz",
    "3b/ord_dataset-3b5db90e337942ea886b8f5bc5e3aa72.pb.gz",
    "3b/ord_dataset-3bcdb559226a40d89406474c02d082d1.pb.gz",
    "3d/ord_dataset-3d470a6df4a04b1996e024a38c53e818.pb.gz",
    "3e/ord_dataset-3e4cbfe3ea40460399482e6cf2f9c894.pb.gz",
    "3e/ord_dataset-3e699bae9dce4a0f996c34a7c5a4b79a.pb.gz",
    "3e/ord_dataset-3e8f24b5bc8e4d8bb9b6e7e89a956e12.pb.gz",
    "3e/ord_dataset-3ec273742a0345ea916ad5fd071167f2.pb.gz",
    "3e/ord_dataset-3ef78abb9a384506b240a419b70e6ceb.pb.gz",
    "3f/ord_dataset-3f2a531ffbae4b9eb1048afcd7953beb.pb.gz",
    "3f/ord_dataset-3f9174c7efcb4f31becbd3516cde9572.pb.gz",
    "3f/ord_dataset-3fa0a6b7d51b4fc6a5380aa0d03ac884.pb.gz",
    "3f/ord_dataset-3feb2a95f66e4706a4a50c977ccd9bf8.pb.gz",
    "40/ord_dataset-406d20cb3f314e2c967b14f55925f895.pb.gz",
    "41/ord_dataset-41558b7e18dd41999311b1762e0e13a3.pb.gz",
    "41/ord_dataset-41760195182e4bb4bc779bd722456071.pb.gz",
    "41/ord_dataset-41c90677d90b4fa5836ab66e2f5b4f51.pb.gz",
    "41/ord_dataset-41ea179beba54bed8cd874a5ec3469f7.pb.gz",
    "42/ord_dataset-4226e9b4f9f845db967ed997270dcafc.pb.gz",
    "42/ord_dataset-42629b4cf1094978a5e5f29f22639ee7.pb.gz",
    "42/ord_dataset-42cd9bc5aae3485f9a58f1527bd3abf0.pb.gz",
    "43/ord_dataset-437aa6654d5044ddaef3346dc4c6e08a.pb.gz",
    "44/ord_dataset-44d518e567bd4c039d77233023f78bb2.pb.gz",
    "45/ord_dataset-45d20d09e4d64f45bdd419044025b4d3.pb.gz",
    "46/ord_dataset-46d216f2c4374ef5b9df90427e2a4cd4.pb.gz",
    "46/ord_dataset-46ff9a32d9e04016b9380b1b1ef949c3.pb.gz",
    "47/ord_dataset-4706e7a7f3cd421bb42b7f877cff8af9.pb.gz",
    "47/ord_dataset-4731a430f0af464b83e8b5b145cd2c77.pb.gz",
    "47/ord_dataset-47bd90bf5ec74fcd99ce250a56e18c8f.pb.gz",
    "48/ord_dataset-488402f6ec0d441ca2f7d6fabea7c220.pb.gz",
    "48/ord_dataset-48929f64ce614f1181a555eafd7c97a6.pb.gz",
    "49/ord_dataset-49124ff635234889bd8dcfe87f4f9013.pb.gz",
    "49/ord_dataset-4937da99a6a247eb90fa70f0d2eac3db.pb.gz",
    "49/ord_dataset-49bd993346b64eeeb2a8fe2643164a0a.pb.gz",
    "4a/ord_dataset-4a4f564f09ba483589e09e154ad32acf.pb.gz",
    "4a/ord_dataset-4a5b4fffffb34daa876fff1f127a4135.pb.gz",
    "4a/ord_dataset-4ac1b977dc504cb5a6a8e85d7d777c45.pb.gz",
    "4a/ord_dataset-4ad5db8537994579bef51f16dd8bf0bd.pb.gz",
    "4b/ord_dataset-4b705442211b4a3988e26d5f65098160.pb.gz",
    "4b/ord_dataset-4bc8addcf9cf4845817557760d62d5b5.pb.gz",
    "4c/ord_dataset-4c84bc8a34e1469aa1b156355c91cdcc.pb.gz",
    "4c/ord_dataset-4c8627b52d564809adb9b494879c07c0.pb.gz",
    "4d/ord_dataset-4d431564f3ef4e9c91d8da5836f4eae6.pb.gz",
    "4d/ord_dataset-4d84abdf99524e0fb6c42ab2a3300790.pb.gz",
    "4e/ord_dataset-4e54080057a44c3887653391e24c90b6.pb.gz",
    "4e/ord_dataset-4e81c470cc3b429faf5e1caa50f70a98.pb.gz",
    "4e/ord_dataset-4e9c2fa02a7544fd839206719263345f.pb.gz",
    "4f/ord_dataset-4f3b2ca6df1d41ef8b5008f1f39da0e2.pb.gz",
    "50/ord_dataset-50c29bf1847c44f4b40bfab5f46fe5fe.pb.gz",
    "50/ord_dataset-50cdc205280641d2a3e264f32908e3d0.pb.gz",
    "50/ord_dataset-50f99930fc41474db226bc80774b38df.pb.gz",
    "52/ord_dataset-520610070b3c4780a03b44c7fcecc28f.pb.gz",
    "52/ord_dataset-5237a168f9214b7ca3db5a1dc3e62d07.pb.gz",
    "52/ord_dataset-52a37d876ddb453e86de0c15fa233d29.pb.gz",
    "53/ord_dataset-530502f8e61e455784f93c5faa45c94b.pb.gz",
    "53/ord_dataset-53cd7fd33c6f4f1c804e187bec94be57.pb.gz",
    "54/ord_dataset-54347fcace774f89850681d6dec8009f.pb.gz",
    "54/ord_dataset-5481550056a14935b76e031fb94b88be.pb.gz",
    "55/ord_dataset-5540e162c09f4c04905ddc8ba9c931c6.pb.gz",
    "55/ord_dataset-55de08a995554f558c25fc43eac62359.pb.gz",
    "55/ord_dataset-55e306db9b6b4befbc22504c12b7113d.pb.gz",
    "56/ord_dataset-56747de2718a4ac5bf061651d1cc9e3e.pb.gz",
    "56/ord_dataset-56a22bc0c3b14f87b9aa3f2fc6488ee7.pb.gz",
    "56/ord_dataset-56a7c92b497c48c59951f153285b4adc.pb.gz",
    "56/ord_dataset-56c07ce5503b46d1805bdf471f8f1c55.pb.gz",
    "56/ord_dataset-56e7a343b17a4d6da818127ceb19589d.pb.gz",
    "58/ord_dataset-58ec5adfcd8648dc9e26ee757d289517.pb.gz",
    "58/ord_dataset-58ec6779628e43e2b3f0972725f262e6.pb.gz",
    "59/ord_dataset-5944a40bb4504bbe8185cfdf2a811d03.pb.gz",
    "59/ord_dataset-59f453c3a3d34a89bfd97b6b8b151908.pb.gz",
    "5a/ord_dataset-5a3d853c53674888a5691dce2e398792.pb.gz",
    "5c/ord_dataset-5c25f386f4664070a72d578feacedf86.pb.gz",
    "5c/ord_dataset-5c4ee54447b84205a10f9c0473172972.pb.gz",
    "5c/ord_dataset-5c8a417a8ba04cf0b7f78b9db9af1d01.pb.gz",
    "5c/ord_dataset-5ca9db262dd24c5a9315cdc8ef055b7e.pb.gz",
    "5d/ord_dataset-5d77a731aa10488794c824ad12021f57.pb.gz",
    "5d/ord_dataset-5df93261afc143c3ae919a57ff4fc1d4.pb.gz",
    "5e/ord_dataset-5e1b2445a3d94ea592ddf2c284118a1e.pb.gz",
    "5e/ord_dataset-5e6956e6e8c24a168866a253f4a66c6c.pb.gz",
    "5e/ord_dataset-5eb2900a93c842ee98f26c305e657b61.pb.gz",
    "5e/ord_dataset-5eb7f2689f4a42eba63ad9e37e49a5cd.pb.gz",
    "5e/ord_dataset-5ebf3d05077a4f7fb91a1cd9bdc504d2.pb.gz",
    "5f/ord_dataset-5fb693db3950403e9ce1a516570153bf.pb.gz",
    "60/ord_dataset-6034127657614f02860ed057b62b882e.pb.gz",
    "60/ord_dataset-60a3e71da3174666a50a61dcfa611a9f.pb.gz",
    "60/ord_dataset-60f3171f0342452f8814e7f294e2be8b.pb.gz",
    "62/ord_dataset-6214af00a7eb47f3887ef21a94320a7e.pb.gz",
    "62/ord_dataset-62e11cab5a6e47d89878616c244e20ef.pb.gz",
    "63/ord_dataset-631d58dab387485c8eb0db4c20a232b7.pb.gz",
    "63/ord_dataset-632f0d9054ce41aba87d4970966c34a6.pb.gz",
    "64/ord_dataset-640d007e35e243a286c8b7dd2b77ac1b.pb.gz",
    "64/ord_dataset-64bb36b111c249e0afcce8c5ad9cdbd0.pb.gz",
    "64/ord_dataset-64e3763e75b242749501bacdc0c8baef.pb.gz",
    "65/ord_dataset-653be8036d754ce7b8a1c4cd419eaf55.pb.gz",
    "65/ord_dataset-65c44df6676d4ce3a1874db5d7958ca9.pb.gz",
    "66/ord_dataset-66f304805e5d47fc8d3c722b1bd8dfa2.pb.gz",
    "67/ord_dataset-67612e25ea9d4b29966a776893a43d59.pb.gz",
    "67/ord_dataset-67c9ee844bf341888b345c8e36183b40.pb.gz",
    "67/ord_dataset-67ed03c283094854909157b1038e38e3.pb.gz",
    "68/ord_dataset-685186618e9f4e7aaa72ac40c16ef354.pb.gz",
    "68/ord_dataset-68715347640045adb1b09e6a04722b0e.pb.gz",
    "68/ord_dataset-68cb8b4b2b384e3d85b5b1efae58b203.pb.gz",
    "6a/ord_dataset-6a0bfcdf53a64c07987822162ae591e2.pb.gz",
    "6b/ord_dataset-6ba93916f2ad4247804469d8c2600b26.pb.gz",
    "6c/ord_dataset-6c36eb0f817d4144988b8963c5d58879.pb.gz",
    "6c/ord_dataset-6c3ec086c8c9475e8d31a44641b49e02.pb.gz",
    "6c/ord_dataset-6cb04513a4a244c0b612b566096f4b3d.pb.gz",
    "6d/ord_dataset-6d529f04a1724b5d85dcc290cf4c38bb.pb.gz",
    "6d/ord_dataset-6d96290c60d941d098c4ddc9d0cb01a0.pb.gz",
    "6d/ord_dataset-6dbd68e494f94ae796187f53698183da.pb.gz",
    "6f/ord_dataset-6f715747ea754945a2f0af4a8482c7d2.pb.gz",
    "70/ord_dataset-70899a0178cc441482746c093624afa0.pb.gz",
    "72/ord_dataset-72d9f7ef86df410fac4765c9632d459b.pb.gz",
    "72/ord_dataset-72fffaae67c8473fb9d951cb1b026646.pb.gz",
    "73/ord_dataset-73916d628db147c89020b3baac642d48.pb.gz",
    "74/ord_dataset-7448b89163bf426c9d9777809ce24cec.pb.gz",
    "74/ord_dataset-744b04e8228742eb9aa4bde36f5dedf1.pb.gz",
    "74/ord_dataset-7456bda2326f4bebaa874a5474d4cc0d.pb.gz",
    "75/ord_dataset-754e10d30fb249229e130865010ab25b.pb.gz",
    "75/ord_dataset-75ca79f43dad4cc8a934fe6487fa8eb1.pb.gz",
    "76/ord_dataset-769fac6048e548eca2d49e48f972884b.pb.gz",
    "76/ord_dataset-76a008eb2d3f48d891cad325041f3d1e.pb.gz",
    "76/ord_dataset-76dd1b78ee414d2da0ed30700ef026f7.pb.gz",
    "77/ord_dataset-7724d657b6fc42848b62ca7c93d84bed.pb.gz",
    "77/ord_dataset-772de90433b44abbbacf9b316b845c31.pb.gz",
    "77/ord_dataset-7774db17e619477ea20ee621abe71257.pb.gz",
    "78/ord_dataset-7860c6f563014da8948ede63b7110bde.pb.gz",
    "78/ord_dataset-78c3f723155a4347a902b53bcee1524d.pb.gz",
    "7a/ord_dataset-7a32ed4947a84db9890d8863ca9e55fb.pb.gz",
    "7a/ord_dataset-7a74d48eeefd45aba53e7258f3ae067a.pb.gz",
    "7a/ord_dataset-7a8649d55889427e85b208ae89475895.pb.gz",
    "7b/ord_dataset-7b02d32cc502407f94aea8e5caf405a2.pb.gz",
    "7b/ord_dataset-7bed824f566d4af0b51d74d386b14bd6.pb.gz",
    "7c/ord_dataset-7c28974b7fcf4c9c86d5f2a42ba328a2.pb.gz",
    "7c/ord_dataset-7c810806c4564bada4a9550135bbb06f.pb.gz",
    "7d/ord_dataset-7d359d96b3a64882921ebdc6c850e22e.pb.gz",
    "7d/ord_dataset-7d8f5fd922d4497d91cb81489b052746.pb.gz",
    "7f/ord_dataset-7f88a5da29d74264aae5239be74b3981.pb.gz",
    "7f/ord_dataset-7fed188163cf4ccca934ec71504c7f8a.pb.gz",
    "80/ord_dataset-8034115bd2ec4d3e95bd3ff7cfde0bde.pb.gz",
    "82/ord_dataset-8214eb8444a44dc2900ccb42dbeff15e.pb.gz",
    "82/ord_dataset-82e842e611ef4a05b6e7f9ea0a46d52d.pb.gz",
    "83/ord_dataset-83acb82dc5ba4f7aba439b9875aaac43.pb.gz",
    "84/ord_dataset-843ef38b45484f72826f5f39d8a29c4d.pb.gz",
    "84/ord_dataset-844a22e1fcab44a5b59c5e2922b2855a.pb.gz",
    "84/ord_dataset-846d411edee44814931e062174d7ef12.pb.gz",
    "84/ord_dataset-84dc0c9e5fb4424687194ef8d14d1c22.pb.gz",
    "85/ord_dataset-8537fa92abf34c849134600c0c2bbcc7.pb.gz",
    "85/ord_dataset-85c00026681b46f89ef8634d2b8618c3.pb.gz",
    "87/ord_dataset-87aeeef7c91d4312a992c5cea936705c.pb.gz",
    "88/ord_dataset-8868acadaee548d5802e504148fc3b63.pb.gz",
    "88/ord_dataset-8881493772e04c4b9495fc2114379967.pb.gz",
    "89/ord_dataset-892acf7477db4d3a8a8559f004a7c0a2.pb.gz",
    "89/ord_dataset-89b083710e2d441aa0040c361d63359f.pb.gz",
    "8a/ord_dataset-8af141577c90485cb9c91079a5ef96b3.pb.gz",
    "8c/ord_dataset-8c74302143c04eb9983e4b3a7ead2d72.pb.gz",
    "8c/ord_dataset-8c94910e81cc4281bd57a58ce657576f.pb.gz",
    "8c/ord_dataset-8ca24382d55b4303bc34393399f4110e.pb.gz",
    "8c/ord_dataset-8cbb58558c904b2b85fa7a1b084a0de9.pb.gz",
    "8c/ord_dataset-8cce6f317d644b348a7978a2dce3ea01.pb.gz",
    "8c/ord_dataset-8cd3720738054d76b936f66e14d8cba6.pb.gz",
    "8d/ord_dataset-8d1e28ec1f6b4ee8ad372e0b5ed7a62e.pb.gz",
    "8d/ord_dataset-8d5c200bca27407ab9febe7598e16458.pb.gz",
    "8e/ord_dataset-8e59bd24817446f7b1c68e805b8e5f1d.pb.gz",
    "8f/ord_dataset-8f61d01cf5b34e85a3ccc89f8901797b.pb.gz",
    "8f/ord_dataset-8fe3c7256e6747e59933c880f5f642a0.pb.gz",
    "90/ord_dataset-90b0aa1f83334a02919b2be3a1c04542.pb.gz",
    "90/ord_dataset-90c12c4a89ae46d88aea27f4e37ba124.pb.gz",
    "91/ord_dataset-912d62c1690b4ebfbe998c0c9baf88a8.pb.gz",
    "92/ord_dataset-9240b22758dd4f9e981f9ad2d5394155.pb.gz",
    "93/ord_dataset-930bf31b476c482e8ba87824089eb600.pb.gz",
    "93/ord_dataset-93908aaae836460ebd48d733eccad483.pb.gz",
    "94/ord_dataset-94e21e9990034c729ea727e7d2ab0eb0.pb.gz",
    "96/ord_dataset-960c6b9c4fc74afd90a3ebf713215626.pb.gz",
    "96/ord_dataset-9625b7c45a574cd58946dd2c1803eb6f.pb.gz",
    "96/ord_dataset-96711be098434bc9ab567cb77fa3362b.pb.gz",
    "97/ord_dataset-9741bb5fd93044078df2a45f45733054.pb.gz",
    "97/ord_dataset-97eb2ab57fec4160922caae33b54d956.pb.gz",
    "99/ord_dataset-993feac5ecf54388aa6326a220e46db3.pb.gz",
    "9a/ord_dataset-9ad0840101374618a7d5c4e4521c82d1.pb.gz",
    "9a/ord_dataset-9adbd9cd2fd941f7af27fb31c9bf3bac.pb.gz",
    "9b/ord_dataset-9b8aa9a7835143ef8ce3f70abfab7545.pb.gz",
    "9c/ord_dataset-9cc455db05a444779921f786a45b21a6.pb.gz",
    "9c/ord_dataset-9cd817a75dfc4fe7ad19d4232772d5ff.pb.gz",
    "9c/ord_dataset-9cecb3a8d3b9494191b28dcefea66af2.pb.gz",
    "9d/ord_dataset-9df8b3ec9c8742b3802e0efaac6f6ef3.pb.gz",
    "9f/ord_dataset-9f54014563f44962b72e288618421b97.pb.gz",
    "a0/ord_dataset-a0071d97083e4e69ae8872417ed2776b.pb.gz",
    "a0/ord_dataset-a0b0a122a04c40798e0535b968adcf6e.pb.gz",
    "a0/ord_dataset-a0bc986c3cf848c2a1ea93b297d1ad89.pb.gz",
    "a0/ord_dataset-a0eff6fe4b4143f284f0fc5ac503acad.pb.gz",
    "a1/ord_dataset-a192df1b44174b5886ef2005f759d553.pb.gz",
    "a1/ord_dataset-a1d4c9f5edd844798727d514977ca73e.pb.gz",
    "a1/ord_dataset-a1e9aa99368e4e5da8b1786b1c05521d.pb.gz",
    "a2/ord_dataset-a20aed058d7b40bc81fdf50bc5b03f97.pb.gz",
    "a2/ord_dataset-a2a0fbbcd49a46ffaa432d7ceae8a506.pb.gz",
    "a2/ord_dataset-a2ae447c4340438c8c3a827109aeb425.pb.gz",
    "a2/ord_dataset-a2d74266062e4398bc26c4f876903ab8.pb.gz",
    "a4/ord_dataset-a495451286334c5c9bbcbd48a00c1350.pb.gz",
    "a4/ord_dataset-a4a191e812a64d0598ea918f047e8da7.pb.gz",
    "a4/ord_dataset-a4f0b79f6b9847168861270b79f84afa.pb.gz",
    "a5/ord_dataset-a5669edbeffe43bf8514c1bfede8f882.pb.gz",
    "a5/ord_dataset-a58d1baeeea441fb9918c10f18f2cdb9.pb.gz",
    "a6/ord_dataset-a6643d22de674f30a85ba57198b82644.pb.gz",
    "a7/ord_dataset-a7baa616c65d42559e25ca0ba61e0744.pb.gz",
    "a7/ord_dataset-a7bd0db0684c464bb02ff6a36065fee3.pb.gz",
    "a8/ord_dataset-a86112d52cd54525a5e36d41f18aced2.pb.gz",
    "a9/ord_dataset-a9b95e50436441ff8f3f12dd60d1e1b2.pb.gz",
    "a9/ord_dataset-a9ba4801408c4b01922886164c10a391.pb.gz",
    "aa/ord_dataset-aa5bc55d09b7465ab353e144a7ac3ad1.pb.gz",
    "aa/ord_dataset-aa9e93ade540499c920a9c3b18e8edaa.pb.gz",
    "aa/ord_dataset-aaeaab5f3720492494c1cbbdd0ed2820.pb.gz",
    "ab/ord_dataset-ab5ba9f863cb41d9924be0bb3c730818.pb.gz",
    "ab/ord_dataset-abe42048f0b84fa88446cee56a31e967.pb.gz",
    "ac/ord_dataset-ac04cf1ba5724e9b93d39b77e9740b21.pb.gz",
    "ac/ord_dataset-ac1c7aa04d6c4e588e723e3e05721681.pb.gz",
    "ac/ord_dataset-ac78456835404910b3a4c840248b6ac9.pb.gz",
    "ac/ord_dataset-acbdbaa766314b66a982823354e82bf2.pb.gz",
    "ad/ord_dataset-ad17798fcea64e26ba91604fca520090.pb.gz",
    "ad/ord_dataset-ad879e603f9440f6bfb7fbd18e5e8761.pb.gz",
    "ae/ord_dataset-aeb5378e6e67443e8a4f5c7a5ec3de51.pb.gz",
    "af/ord_dataset-af85e6f81c2d49f08086afd6d9e6959c.pb.gz",
    "af/ord_dataset-afd812677c134591a99f46ce28de2524.pb.gz",
    "b0/ord_dataset-b0ddd49dad024fc7a23ae6f474f9c52f.pb.gz",
    "b1/ord_dataset-b1023e5ccd7142de9d250aa2e3e124db.pb.gz",
    "b1/ord_dataset-b17fdf55f5f945a48d47a588fc255573.pb.gz",
    "b1/ord_dataset-b18df02d6e9345faa0f2dae281a0870a.pb.gz",
    "b1/ord_dataset-b195433d5c354ddfb6cde0d53c41910f.pb.gz",
    "b1/ord_dataset-b1a34bc8c1204d51a772ed27396c794e.pb.gz",
    "b1/ord_dataset-b1bf0950b5cb430abbb1d80744f5df84.pb.gz",
    "b2/ord_dataset-b2a00a09c8494aaf9348c3a3af29d26e.pb.gz",
    "b3/ord_dataset-b37811c5ed724c3b844cc4d2b5640398.pb.gz",
    "b4/ord_dataset-b43eb0158fd54801957aaa07bbdf3057.pb.gz",
    "b4/ord_dataset-b440f8c90b6343189093770060fc4098.pb.gz",
    "b5/ord_dataset-b5f1497361c94e5fa55257200189a6a4.pb.gz",
    "b6/ord_dataset-b6309a86b70f4ca08fd301828cacf950.pb.gz",
    "b6/ord_dataset-b67d30cd146f49dcbf24faee022f1a09.pb.gz",
    "b6/ord_dataset-b6d8835b0c934476a36e6149e7597487.pb.gz",
    "b7/ord_dataset-b76b52f4448a4eedb28ffcd8f902046a.pb.gz",
    "b8/ord_dataset-b832171fa1674c469fbd33bc0ca1d5f1.pb.gz",
    "b8/ord_dataset-b83b16f2fb1a4f8782f3d82c1766cc6b.pb.gz",
    "b8/ord_dataset-b8b98725045d45bdbd73512048f4b47e.pb.gz",
    "b9/ord_dataset-b90b48a6c23149a1aa2d91b92b1a31e4.pb.gz",
    "b9/ord_dataset-b971427c0b944c56b63bb2356fa8ca69.pb.gz",
    "b9/ord_dataset-b9a9e369e9da4413999591aa08f4c3e3.pb.gz",
    "b9/ord_dataset-b9d8ddf9c2884d40859b6b5f24815a7d.pb.gz",
    "ba/ord_dataset-ba1248d90e3e465693710bbc45866f37.pb.gz",
    "ba/ord_dataset-ba7561dae3884c07a8beddd0b9f1222e.pb.gz",
    "bb/ord_dataset-bb4579c57ddc48c6a01fad97dacb6293.pb.gz",
    "bb/ord_dataset-bbd7e53f000345838ad4920a07a169ff.pb.gz",
    "bc/ord_dataset-bcc0b01d4f58457a8733b10a099f43ba.pb.gz",
    "bd/ord_dataset-bd998d3fe946475e835418aaf647c00d.pb.gz",
    "bd/ord_dataset-bdb961f26fac426eaa2de8f54a284acf.pb.gz",
    "be/ord_dataset-be22f1f5114b4ebfb8d447c93d62d62c.pb.gz",
    "be/ord_dataset-be508e976bbb4586a0cc3368302a62f8.pb.gz",
    "be/ord_dataset-be83cbc722064f3696975001242f9f1a.pb.gz",
    "be/ord_dataset-becaf7dc22c44672b22e4b94335cbf08.pb.gz",
    "bf/ord_dataset-bf316bf78f4f45d4b275959c08104b7c.pb.gz",
    "bf/ord_dataset-bfcf5a01f1a04ec585ba3f28cb93c8c9.pb.gz",
    "c1/ord_dataset-c1e70ad912eb438f8d34b1dc681f809a.pb.gz",
    "c2/ord_dataset-c2ad1656a3ca4d08888ffb6e3f3a2742.pb.gz",
    "c3/ord_dataset-c31cb9b44c404c10ba3aa533aa079e2b.pb.gz",
    "c3/ord_dataset-c34472859da14a81bfe0f74e60f15c43.pb.gz",
    "c3/ord_dataset-c3a75813d0b24864aa4f7cd526efd6aa.pb.gz",
    "c3/ord_dataset-c3c1091f873b4f40827973a6f1f9b685.pb.gz",
    "c4/ord_dataset-c408dfed796e4354b61e312e67f7143f.pb.gz",
    "c5/ord_dataset-c544c0c663f54dbea4ddb52ddde7934e.pb.gz",
    "c5/ord_dataset-c5ee194443334d3e92aff17e46e33bd1.pb.gz",
    "c6/ord_dataset-c663259b80f947e2a8923796fb0e9a6b.pb.gz",
    "c8/ord_dataset-c8069773c1a148aca8ab417108daacc5.pb.gz",
    "c8/ord_dataset-c8a367b56b4f406b878f51867b157d19.pb.gz",
    "c8/ord_dataset-c8db783fb93842e89cdcd41afa59a141.pb.gz",
    "c9/ord_dataset-c975a50a7600448fabd558f4a94a3e29.pb.gz",
    "c9/ord_dataset-c9f990dde2dc45d0948ecbe037a0d819.pb.gz",
    "ca/ord_dataset-cac8df8aff894288876df4e093c9877f.pb.gz",
    "cb/ord_dataset-cb0fdfa937234c78b020ec878a93b94c.pb.gz",
    "cb/ord_dataset-cb5c1a9eddff4790b1bd650617c32d34.pb.gz",
    "cb/ord_dataset-cb5dd7a8b94e4f19a9148a1904b0dcb6.pb.gz",
    "cb/ord_dataset-cb9f452f8418479ab5fd99c835dbe015.pb.gz",
    "cb/ord_dataset-cbcc4048add7468e850b6ec42549c70d.pb.gz",
    "cc/ord_dataset-cc0899cd744f4f7f8e7f2463560faad1.pb.gz",
    "cd/ord_dataset-cd531114850e4f239b2a3661044ae672.pb.gz",
    "cd/ord_dataset-cde802cdb7434a5f82a22981ccaefc4e.pb.gz",
    "ce/ord_dataset-ce71a906ea9c4399a2014cbaaff88c8f.pb.gz",
    "cf/ord_dataset-cf82113c6dd341d4a014b4667994f01c.pb.gz",
    "cf/ord_dataset-cfad8b3f00044bcda60a96b019f09872.pb.gz",
    "d0/ord_dataset-d0003732f14e4648a00d341275654590.pb.gz",
    "d0/ord_dataset-d06b137b66b3478fb6a0d41a5efd32f8.pb.gz",
    "d1/ord_dataset-d182ca8cf3f34de69c7a38343db8c930.pb.gz",
    "d1/ord_dataset-d1bd8c96676b4d21aad27b173c6b4eff.pb.gz",
    "d1/ord_dataset-d1c545e3afc447099315419421478aab.pb.gz",
    "d2/ord_dataset-d23f2f15886d4c22b7d7ba5f0e203e81.pb.gz",
    "d2/ord_dataset-d24cc23b3db94284bb83a2ccdd9eb880.pb.gz",
    "d2/ord_dataset-d26118acda314269becc35db5c22dc59.pb.gz",
    "d3/ord_dataset-d31180f42ced44719fd9e72685c798bf.pb.gz",
    "d3/ord_dataset-d319c2a22ecf4ce59db1a18ae71d529c.pb.gz",
    "d3/ord_dataset-d330662edaed408d950898172aba7a4c.pb.gz",
    "d3/ord_dataset-d3580a12b15e46cc91aa73f4f1a4a8cc.pb.gz",
    "d5/ord_dataset-d56f0a7ec215495c92e641d9fa932d28.pb.gz",
    "d5/ord_dataset-d5bb2294ac964841b8ffc9e0a34e93af.pb.gz",
    "d5/ord_dataset-d5c54236ecd94d61aaa071461bcfc426.pb.gz",
    "d6/ord_dataset-d625331704b34593871e69cd09a5cd83.pb.gz",
    "d6/ord_dataset-d63ab258f4634f87bdebbaff0a341c28.pb.gz",
    "d6/ord_dataset-d673d02cdac14dba9ff59f12845a4f37.pb.gz",
    "d6/ord_dataset-d6752864fa634eb3b05f7440b4fb9a70.pb.gz",
    "d6/ord_dataset-d6b60b593b1c4668bc6843cd65a5d232.pb.gz",
    "d6/ord_dataset-d6cdba90760a47779a36ece5962905eb.pb.gz",
    "d7/ord_dataset-d728a2f811c0424cbcdb5a84d02b93ae.pb.gz",
    "d8/ord_dataset-d8a5dc784dde4465894ec7c69d2e3ba6.pb.gz",
    "d9/ord_dataset-d9235b0d1f9f4e85b2690ee8f860dc30.pb.gz",
    "d9/ord_dataset-d932d1d683704a8bad3d064bcb197acc.pb.gz",
    "d9/ord_dataset-d986d6c6630b41ee8d401fd85be47701.pb.gz",
    "d9/ord_dataset-d98d003e9abb4b579746c5a361466e14.pb.gz",
    "d9/ord_dataset-d9fb402fe3e745c1893a29161cc5dda2.pb.gz",
    "da/ord_dataset-da49b0378abf41bf92ab8ecdd3feb28b.pb.gz",
    "da/ord_dataset-dabcd66611f34094b1baca0095305a66.pb.gz",
    "db/ord_dataset-dba7c8cf38834984ba5e6376158f46fe.pb.gz",
    "dc/ord_dataset-dc3bb1b1ac4e4229a3fc28fb559a9777.pb.gz",
    "dc/ord_dataset-dc78376bd0a946d1a5a41eec30282b7d.pb.gz",
    "dd/ord_dataset-dd1de64954674e17b4b690cac09dc67b.pb.gz",
    "dd/ord_dataset-dd320ded4b3f4764af39de99491533f7.pb.gz",
    "dd/ord_dataset-ddb1b60bec5c45e9a2938b6dac04376c.pb.gz",
    "de/ord_dataset-de0979205c84441190feef587fef8d6d.pb.gz",
    "de/ord_dataset-de10a15943a54ac7a3c3e1c774d21392.pb.gz",
    "de/ord_dataset-de283386b8034acd99fba96d3c7d3227.pb.gz",
    "de/ord_dataset-de51ecc8d4434bacaa8bc32d7d73484c.pb.gz",
    "de/ord_dataset-de6bce51790e4004a27e1a8f2bcc7ded.pb.gz",
    "e0/ord_dataset-e0a818f9350b46cdb184d2ac404ede9f.pb.gz",
    "e1/ord_dataset-e1c3af9b105b4af09a5171403bbfc06f.pb.gz",
    "e2/ord_dataset-e2b35e721c2741999b0005d12691f9fe.pb.gz",
    "e4/ord_dataset-e402405fd21e4770a14f157cb62ca439.pb.gz",
    "e4/ord_dataset-e44331dc51de453ca14b7032593c1958.pb.gz",
    "e5/ord_dataset-e5870fef54544989915ccced8dd8d849.pb.gz",
    "e6/ord_dataset-e61c1950b2fc468dbf0701c65768f73f.pb.gz",
    "e6/ord_dataset-e6e5ffd5405c481d8061087049155f9f.pb.gz",
    "e7/ord_dataset-e7e9fe3dd42b4f499d5b17fe63bbf336.pb.gz",
    "e8/ord_dataset-e8c6a25568b64529b960953990e6921f.pb.gz",
    "e9/ord_dataset-e90cd41afe844e49875435eb99903799.pb.gz",
    "e9/ord_dataset-e967d076b4894c2c854795f019ed3c39.pb.gz",
    "e9/ord_dataset-e96f5a2842f14e5380461c234100f05a.pb.gz",
    "e9/ord_dataset-e984b2d4813f44d59867e1771acc9b66.pb.gz",
    "ea/ord_dataset-eacfee6d16d8455a93348409f1b37be4.pb.gz",
    "eb/ord_dataset-eb32c2b9cbdf4fc39ce6c9fe0fa11430.pb.gz",
    "eb/ord_dataset-eb4226b4f7644a01a737e7547b70014a.pb.gz",
    "ec/ord_dataset-ec576c604a9d47258c87c732a043ec71.pb.gz",
    "ec/ord_dataset-ec58fad8331a42c5a67ad75aac6713b4.pb.gz",
    "ec/ord_dataset-ec7cb3d5a8704f64b01d401ea555974f.pb.gz",
    "ec/ord_dataset-ec9decb576c4424c9685993f6262bd9c.pb.gz",
    "ed/ord_dataset-ed5a3d1f8dc744759e7fa1c320a44e59.pb.gz",
    "ed/ord_dataset-ed65749688da45af8a8432967b017729.pb.gz",
    "ed/ord_dataset-ed680843f6d14f5c9901869b2a06b4a4.pb.gz",
    "ed/ord_dataset-ed79ebfb2fff4cdbbc3a609c8edeac11.pb.gz",
    "ed/ord_dataset-ed846b4b229e4bb09ad9dba95ad7ebeb.pb.gz",
    "ee/ord_dataset-ee287d49cb8642e59ae9c3951f746312.pb.gz",
    "ee/ord_dataset-ee5599340390470d8e5b5ac1feddf9d6.pb.gz",
    "ee/ord_dataset-eeba974d3c284aed86d1c1d442260a1e.pb.gz",
    "ee/ord_dataset-eed8d32c2f1d46a89ba39d04dfaa83b1.pb.gz",
    "f0/ord_dataset-f024e9664ab64906a71a2ff6004cb3d0.pb.gz",
    "f0/ord_dataset-f027aa93238e424fbbf9bad1c7699adc.pb.gz",
    "f0/ord_dataset-f0794d5ded7a4d3fab45cc3d7a89ee3d.pb.gz",
    "f0/ord_dataset-f0d0fe3f599c471ca1c011dbbcdeeff2.pb.gz",
    "f1/ord_dataset-f196f0a87dd74fcd82ca019f8ff5cf9c.pb.gz",
    "f1/ord_dataset-f1b9e9741a6a4cc68e7070df811f86bb.pb.gz",
    "f2/ord_dataset-f222e615b1f74f0fabef9cd9b98516b7.pb.gz",
    "f2/ord_dataset-f25e1b7f8ef54305a5170f5395a768c7.pb.gz",
    "f3/ord_dataset-f367d8d2baac490b9204609a79420961.pb.gz",
    "f3/ord_dataset-f3b9c7d90d3648ff8136643f634ef2f0.pb.gz",
    "f4/ord_dataset-f4512fe8cd804ac79da66cbc0c2b9d42.pb.gz",
    "f4/ord_dataset-f483e698250b4da0a84f425c7bfa965a.pb.gz",
    "f5/ord_dataset-f5d908a6dcb44353a706166b5e9f7f88.pb.gz",
    "f5/ord_dataset-f5dc3e448c204058b116bf1970695bef.pb.gz",
    "f6/ord_dataset-f6fafbb8ce5f4ef099be3a772075ec97.pb.gz",
    "f8/ord_dataset-f886e51ba1484c76a94bce1482f1eab9.pb.gz",
    "f8/ord_dataset-f8e6e6a2d2bf4135b9e346456c81700f.pb.gz",
    "fa/ord_dataset-fa3b512e2d924b9b965301ebcba6853d.pb.gz",
    "fa/ord_dataset-faa0236be76c4501841c954527cd1b6c.pb.gz",
    "fb/ord_dataset-fb70ed83140e4f53a907d87192ad748c.pb.gz",
    "fb/ord_dataset-fb72428f30234761b4216139dc228d0c.pb.gz",
    "fb/ord_dataset-fbdd058349aa456f812e3546c84baab5.pb.gz",
    "fc/ord_dataset-fc4cd2699fc04ecbb7c7c831849e6b22.pb.gz",
    "fc/ord_dataset-fc83743b978f4deea7d6856deacbfe53.pb.gz",
    "fc/ord_dataset-fcf7c02bbb814fe696760b5fbfee16bb.pb.gz",
    "fd/ord_dataset-fd1553bae35046c7a67523ff472cb5c3.pb.gz",
    "fd/ord_dataset-fd1fa959d6264608b0b7fcda16741bfd.pb.gz",
    "fd/ord_dataset-fd44e5eeeeb4473cb5fbff2d885d7833.pb.gz",
    "fd/ord_dataset-fdef1f30cad64430bf05cf108d8004a2.pb.gz",
    "fe/ord_dataset-fe016e2f90e741a590ad77fd5933161f.pb.gz",
    "fe/ord_dataset-fea3ada7aaad45a0b4e7922640986af3.pb.gz",
    "ff/ord_dataset-ff0bcb6c2300494cbdf16005ad32ad5f.pb.gz",
    "ff/ord_dataset-ffbef48837674f39816de887b5dc8bae.pb.gz",
]
_CHUNKS = len(_FILES)

_CITATION = """\
@article{kearnes2021open,
  title={The open reaction database},
  author={Kearnes, Steven M and Maser, Michael R and Wleklinski, Michael and Kast, Anton and Doyle, Abigail G and Dreher, Spencer D and Hawkins, Joel M and Jensen, Klavs F and Coley, Connor W},
  journal={Journal of the American Chemical Society},
  volume={143},
  number={45},
  pages={18820--18826},
  year={2021},
  publisher={ACS Publications}
}
"""

_DESCRIPTION = """\
Chemical reaction datasets from the Open Reaction Database
"""

_HOMEPAGE = "https://github.com/open-reaction-database/ord-data"

_LICENSE = "CC-BY-SA-4.0"


class OrdDataset(datasets.GeneratorBasedBuilder):
    """Open Reactions Database"""

    VERSION = datasets.Version("0.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="full", version=VERSION, description="Complete dataset"
        ),
        datasets.BuilderConfig(
            name="small", version=VERSION, description="Small version"
        ),
    ]

    DEFAULT_CONFIG_NAME = "full"

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "rxn_id": datasets.Value("string"),
                "rxn_name": datasets.Value("string"),
                "yield": datasets.Value("float32"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # picked by rolling fair die
        test_chunks = list(range(len(_FILES) - 1, len(_FILES) - 6, -1))
        val_chunks = list(range(len(_FILES) - 6, len(_FILES) - 11, -1))
        train_chunks = [
            i for i in range(_CHUNKS) if i not in val_chunks and i not in test_chunks
        ]
        if self.config.name == "small":
            train_chunks = train_chunks[:1]
            val_chunks = val_chunks[:1]
            test_chunks = test_chunks[:1]
        urls = {
            "test": [f"{_URL}{_FILES[i]}?raw=True" for i in test_chunks],
            "dev": [f"{_URL}{_FILES[i]}?raw=True" for i in val_chunks],
            "train": [f"{_URL}{_FILES[i]}?raw=True" for i in train_chunks],
        }
        paths = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepaths": paths["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepaths": paths["dev"],
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepaths": paths["test"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepaths, split):
        key = 0
        for path in filepaths:
            with datasets.streaming.xopen(path, "rb") as f:
                data = dataset_pb2.Dataset.FromString(f.read())
            for i in range(len(data.reactions)):
                # rxn_id = data.reaction_ids[i] ?? index errors
                rxn = data.reactions[i]
                reactants = ""
                reagents = ""
                products = ""
                rxn_yield = None
                rxn_name = None
                failed = False
                for k, v in rxn.inputs.items():
                    for c in v.components:
                        if c.reaction_role == reaction_pb2.ReactionRole.REACTANT:
                            try:
                                reactants += next(
                                    i.value
                                    for i in c.identifiers
                                    if i.type == reaction_pb2.CompoundIdentifier.SMILES
                                )
                                reactants += "."
                            except StopIteration:
                                failed = True
                        elif (
                            c.reaction_role == reaction_pb2.ReactionRole.SOLVENT
                            or c.reaction_role == reaction_pb2.ReactionRole.REAGENT
                            or c.reaction_role == reaction_pb2.ReactionRole.CATALYST
                        ):
                            try:
                                reagents += next(
                                    i.value
                                    for i in c.identifiers
                                    if i.type == reaction_pb2.CompoundIdentifier.SMILES
                                )
                                reagents += "."
                            except StopIteration:
                                pass

                if len(reactants) == 0 or failed:
                    continue  # failed
                else:
                    reactants = reactants[:-1]  # remove dot
                if len(reagents) > 0:
                    reagents = reagents[:-1]  # remove dot
                # no idea what we do with more than one outcome yet
                for p in rxn.outcomes[0].products:
                    if p.reaction_role == reaction_pb2.ReactionRole.PRODUCT:
                        try:
                            products += next(
                                i.value
                                for i in p.identifiers
                                if i.type == reaction_pb2.CompoundIdentifier.SMILES
                            )
                            products += "."
                        except StopIteration:
                            failed = True
                        if rxn_yield is None or p.is_desired_product:
                            try:
                                m = next(
                                    m
                                    for m in p.measurements
                                    if m.type == reaction_pb2.ProductMeasurement.YIELD
                                )
                                rxn_yield = m.percentage.value
                            except StopIteration:
                                pass
                if len(products) == 0 or failed:
                    continue  # failed
                else:
                    products = products[:-1]  # remove dot
                # get name
                try:
                    rxn_name = next(
                        i.value
                        for i in rxn.identifiers
                        if i.type == reaction_pb2.ReactionIdentifier.REACTION_TYPE
                    )
                except StopIteration:
                    pass
                yield key, {
                    "text": reactants + ">" + reagents + ">" + products,
                    "rxn_name": rxn_name,
                    "yield": rxn_yield,
                    "rxn_id": key,
                }
                key += 1
