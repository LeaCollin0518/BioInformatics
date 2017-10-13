public class NeedlemanWunsch {
	public static void main(String[] args){
		//parameters
		String s = "";
		String t = "";
		int m = s.length() +1;
		int n = t.length() +1;
		int gap = -2;
		int transition = -1;
		int transversion = -2;
		int match = 1;
		int[][] score = new int[m][n];
		String [][] traceback = new String[m][n];
		int finalScore = 0;
		String alignS = "";
		String alignT = "";
		
		//initializing score matrix
		for(int j=0; j<n; j++){
			score[0][j] = gap*j;
		}
		
		for(int i=0; i<m; i++){
			score[i][0] = gap*i;
		}
		
		//initializing traceback matrix
		for(int j=0; j<n; j++){
			traceback[0][j] = "L";
		}
		
		for(int i=0; i<m; i++){
			traceback[i][0] = "U";
		}
		
		traceback[0][0] = null;
		
		//filling in score matrix
		for(int i=1; i<m; i++){
			for(int j=1; j<n; j++){
				int max =  Integer.MIN_VALUE;
				int matched = 0;
				if(s.charAt(i-1) == t.charAt(j-1)){
					matched = score[i-1][j-1] + match;
				}
				else if((s.charAt(i-1) == 'A' && t.charAt(j-1) == 'G') || (s.charAt(i-1) == 'G' && t.charAt(j-1) == 'A') || (s.charAt(i-1) == 'C' && t.charAt(j-1) == 'T') || (s.charAt(i-1) == 'T' && t.charAt(j-1) == 'C')){
					matched = score[i-1][j-1] + transition;
				}
				else{
					matched = score[i-1][j-1] + transversion;
				}
				int left = score[i][j-1] + gap;
				int up = score[i-1][j] + gap;
				
				if (matched > max){
					max = matched;
				}
				if (left > max){
					max = left;
				}
				if (up > max){
					max = up;
				}
				
				if (max == matched){
					score[i][j] = matched;
					traceback[i][j] = "D";
				}
				else if (max == left){
					score[i][j] = left;
					traceback[i][j] = "L";
				}
				else{
					score[i][j] = up;
					traceback[i][j] = "U";
				}
			}
		}
		
		//getting score of alignment
		finalScore = score[m-1][n-1];
		System.out.println("The optimal alignment has score: " + finalScore);
		
		//tracing back to get actual alignment
		int row = m-1;
		int col = n-1;
		while(row != 0 || col !=0){
				if (traceback[row][col] == "D"){
					alignS = s.charAt(row-1) + alignS;
					alignT = t.charAt(col-1) + alignT;
					row--;
					col--;
				}
				else if(traceback[row][col] == "L"){
					alignS = '-' + alignS;
					alignT = t.charAt(col-1) + alignT;
					col--;
				}
				else if(traceback[row][col] == "U"){
					alignS = s.charAt(row-1) + alignS;
					alignT = '-' + alignT;
					row--;
				}
		}
		System.out.println(alignS);
		System.out.println(alignT);
	}
}
