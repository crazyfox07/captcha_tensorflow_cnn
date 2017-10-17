package test;
 import java.awt.Color;
 import java.awt.Font;
 import java.awt.Graphics;
 import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Date;
import java.util.Random;

 import javax.imageio.ImageIO;
 import javax.servlet.ServletOutputStream;
 import javax.servlet.http.HttpServletRequest;
 import javax.servlet.http.HttpServletResponse;
 import javax.servlet.http.HttpSession;

 public class CaptchaDemo {
	 
	 private static Random random = new Random();
	 //横向扭曲图片
	 private static void shearX(Graphics g, int w1, int h1) {
		 	
		 int period = random.nextInt(6); 
		 
		  boolean borderGap = true; 
		  int frames = 1; 
		  int phase = random.nextInt(10); 
		 
		  for (int i = 0; i < h1; i++) { 
		   double left_right = (double) (period >> 1) 
		     * Math.sin((double) i / (double) period 
		       + (6.2831853071795862D * (double) phase) 
		       / (double) frames); 		  
		   //将(0,i,0+w1,i+1)的区域左右平移(a,0)
		   g.copyArea(0, i, w1, 1,  (int)left_right, 0); 
		   if (borderGap) { 
		    
		    g.drawLine((int) left_right, i, 0, i); 
		    g.drawLine((int) left_right + w1, i, w1, i); 
		   } 
		  } 
		}
	//纵向扭曲图片
	 private static void shearY(Graphics g, int w1, int h1) {

		 int period = random.nextInt(8)+8;		 
		  boolean borderGap = true; 
		  int frames = 20; 
		  int phase = 7; 
		  for (int i = 0; i < w1; i++) { 
		   double up_down = (double) (period >> 1) 
		     * Math.sin((double) i / (double) period 
		       + (6.2831853071795862D * (double) phase) 
		       / (double) frames); 
		   //将(i,0,i+1,0+h1)的区域上下平移(a,0)
		   g.copyArea(i, 0, 1, h1, 0, (int)up_down); 
		   if (borderGap) { 		   
		    g.drawLine(i, (int) up_down, i, 0); 
		    g.drawLine(i, (int) up_down + h1, i, h1); 
		   } 
		 
		  } 		 

		}

 	public static void genCaptcha() throws IOException {
// 		char[] codeSequence = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
//   	            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
//   	            'X', 'Y', 'Z','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
//   	            'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
//   	            'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };
 		String s="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZMXWZ";
 		char[] codeSequence=s.toCharArray();
 		int X_=26;//到图片左方的距离
 		int Y_=35;//到图片右方的距离
 		int space=22;//字间隔；
		int Font_size=37;
 		int W=150;//图片宽度
 		int H=40;//图片高度
 		int[] rand={-1,0,1,2,-2};//随机数选取，微调
 		int num_of_disturb_lines=40;//干扰下条数
 		int num_of_imgs=100000;//生成图片的张数
 		//String font_type="dejavu sans";//字体
 		String[] font_types={"dejavu sans","candara","calibria","tahoma"};

 		for(int n=0;n<num_of_imgs;n++){

 		// 定义图像buffer
 	 		BufferedImage buffImg = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
 	 		Graphics gd = buffImg.getGraphics();
 	 		// 创建�?个随机数生成器类
 	 		Random random = new Random();
 	 		int disturb_lines=num_of_disturb_lines+random.nextInt(10);
 	 		// 将图像填充为白色
 	 		gd.setColor(Color.decode("#FFFFFF"));
 	 		gd.fillRect(0, 0, W, H);
 	 		// 创建字体，字体的大小应该根据图片的高度来定�??
 	 		String font_type=font_types[random.nextInt(4)];
 	 		Font font = new Font(font_type,Font.PLAIN, Font_size+rand[random.nextInt(5)]);
 	 		// 设置字体�?
 	 		gd.setFont(font);
 	 		// 画边框�??
 	 		gd.setColor(Color.decode("#FFFFFF"));
 	 		gd.drawRect(0, 0, W - 1, H - 1);
 	 		gd.setColor(Color.decode("#000000"));
 	 		// 随机产生40条干扰线，使图象中的认证码不易被其它程序探测到�??
 	 		for (int i = 0; i < disturb_lines; i++) {
 	 			int x = random.nextInt(W);
 	 			int y = random.nextInt(H);
 	 			int xl = random.nextInt(12);
 	 			int yl = random.nextInt(12);
 	 			gd.drawLine(x, y, x + xl, y + yl);
 	 		}

 	 		// randomCode用于保存随机产生的验证码，以便用户登录后进行验证�?
 	 		StringBuffer randomCode = new StringBuffer();
 	 		int red = 0, green = 0, blue = 0;

 	 		// 随机产生codeCount数字的验证码�?
 	 		for (int i = 0; i < 4; i++) {
 	 			// 得到随机产生的验证码数字�?
 	 			String code = String.valueOf(codeSequence[random.nextInt(codeSequence.length)]);
 	 			// 产生随机的颜色分量来构�?�颜色�?�，这样输出的每位数字的颜色值都将不同�??

 	 			red = random.nextInt(255);
 	 			green = random.nextInt(255);
 	 			blue = random.nextInt(255);

 	 			// 用随机产生的颜色将验证码绘制到图像中�?
 	 			gd.setColor(new Color(red, green, blue));
 	 			//第一个字距离图片左边26
 	 			if(i==0){
 	 				gd.drawString(code, X_, Y_);
 	 			}
 	 			//最后一个字距离图片右边30+30
 	 			else if(i==3){
 	 				gd.drawString(code, W-29-space+rand[random.nextInt(5)], Y_);
 	 			}
 	 			else{
 	 				gd.drawString(code, X_+space*(i)+rand[random.nextInt(5)], Y_);
 	 			}
				//扭曲图片
 	 			shearX(gd, W, H);//横向扭曲
 	 			//shearY(gd, W, H);//纵向扭曲

 	 			// 将产生的四个随机数组合在�?起�??
 	 			randomCode.append(code);
 	 		}

 	 		String img_name=Long.toString(new Date().getTime())+"_"+randomCode+".png";
 	 		ImageIO.write(buffImg,"png",new File("D:\\tmp\\img\\"+img_name));
 		}

 	}

 	public static void main(String[]args) throws IOException{
 		genCaptcha();

 	}
 }
