LATEX_MACRO = r"""
% ---------------- GLOBAL STYLE ----------------
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.15}
\newcommand{\vqafont}{\footnotesize}
\newcommand{\correct}[1]{\textbf{#1} \checkmark}

% ---------------- ATOMIC VQA BLOCK ----------------
% Args:
% 1 = image path
% 2 = question
% 3..6 = answers A..D
\newcommand{\vqaBlock}[6]{%
\begin{subfigure}{\linewidth}
  \centering
  \includegraphics[width=\linewidth,keepaspectratio]{#1}
  \vspace{0.4em}

  {\vqafont
  \begin{tabular}{p{0.94\linewidth}}
    \centering \textbf{Q:} #2 \\[0.4em]
    \begin{tabular}{p{0.45\linewidth} p{0.45\linewidth}}
      \textbf{A.} #3 & \textbf{B.} #4 \\
      \textbf{C.} #5 & \textbf{D.} #6 \\
    \end{tabular}
  \end{tabular}
  }
\end{subfigure}
}
"""